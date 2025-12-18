"""
This script trains NT to recognize [CLS] as a label and reconstruct the sequence,
possibly using [CLS] and a few [MASK] tokens.
"""

import torch
import wandb
import os
import numpy as np # for plot
import matplotlib.pyplot as plt # for plot

from datasets import load_from_disk, load_dataset
from dataset import DiffusionNTDataset
from config import TrainerNTDiffusionCFG, DiffusionNTCFG, LinearAlpha
from model import DiffusionNT
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "zhihan1996/DNABERT-2-117M" # DNABERT path

def reinit_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class TrainerNTDiffusion:

    def __init__(self, trainer_cfg: TrainerNTDiffusionCFG):

        self.cfg = trainer_cfg
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        ).to(device)

        if trainer_cfg.reinit:
            self.model.apply(reinit_weights)

        # Losses
        self.MLM_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none") # required for MLM

        # Alpha for Diffusion process
        self.alpha = LinearAlpha()

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=trainer_cfg.lr,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=trainer_cfg.num_epochs
        )

        # Dataset
        if trainer_cfg.dataset_path.startswith("dataset"):
            ds = load_from_disk(
                trainer_cfg.dataset_path
            )

            self.train_dataset = ds.filter(
                lambda chrom: chrom in trainer_cfg.train_chroms,
                input_columns=["chrom"]
            )
            self.test_dataset = ds.filter(
                lambda chrom: chrom in trainer_cfg.test_chroms,
                input_columns=["chrom"]
            )

        else:
            # Load from huggingface
            ds = load_dataset(
                trainer_cfg.dataset_path,
                "EPI_K562"
            )
            self.train_dataset = ds["train"]
            self.test_dataset = ds["test"]

        # Dict for losses plots
        self.train_MLM_loss = {}
        self.test_MLM_loss = {}
        self.token_accuracy = {}

        # Init wandb
        self.step = 0
        self.current_epoch = 0
        self.p_start = 0.15
        self.p_end = 0.9
        if trainer_cfg.use_wandb:
            wandb.init(
                project=trainer_cfg.wandb_project,
                name=trainer_cfg.wandb_name
            )
    
    def collate_fn(self, batch, debug: bool = False) -> dict[str, torch.Tensor]:
        
        # Tokenize the sequences
        sequences = [sample["sequence"] for sample in batch]

        tok_out = self.tokenizer(
            sequences,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )

        # Extract ids and attention mask
        batch_input_ids = tok_out["input_ids"]
        batch_attention = tok_out["attention_mask"]

        # Get input shape
        B, L = batch_input_ids.shape

        # Clone input ids for the labels
        labels = batch_input_ids.clone()

        # Create decreasing probability distribution
        # progress = self.current_epoch / self.cfg.num_epochs
        # p = self.p_start + progress * (self.p_end - self.p_start)
        # p = self.p_end - progress * (self.p_end - self.p_start)
        # p = torch.empty(1).uniform_(0.1, 0.9).item() 

        # Create different ps per batch
        # p = torch.empty(B).uniform_(self.p_start, self.p_end).unsqueeze(-1) # unsqueeze for later broadcasting

        # Sample t ~ Uniform[0, 1)
        t = torch.empty(B, 1).uniform_(self.p_start, self.p_end)
        alpha_t = self.alpha(t)
        p = 1 - alpha_t

        # Identify special tokens
        is_special = torch.zeros_like(batch_input_ids, dtype=torch.bool)
        for sid in self.tokenizer.all_special_ids:
            is_special |= (batch_input_ids == sid)

        # Determine which tokens can be masked
        mask_candidate = (batch_attention == 1) & (~is_special)

        # Randomly select tokens to mask based on p
        rand = torch.rand_like(batch_input_ids, dtype=torch.float)
        mask_positions = (rand < p) & mask_candidate

        # Apply masking
        batch_input_ids[mask_positions] = self.tokenizer.mask_token_id

        # Set unmasked positions to -100 in labels (ignored by loss)
        labels[~mask_positions] = -100

        if debug:
            print("=== DEBUG COLLATE ===")
            print("Masked IDs:\n", batch_input_ids)
            print("Labels:\n", labels)
            print("Mask positions:\n", mask_positions)
            print("=====================\n")

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "MLM_labels": labels,
            "t": t # to keep track of timesteps
        }        
    
    def train_dataloader(self):
        return DataLoader(
            DiffusionNTDataset(self.train_dataset, self.cfg.col_name), 
            batch_size=self.cfg.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            DiffusionNTDataset(self.test_dataset, self.cfg.col_name),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
   
    def train_step(self, input_ids, attention_mask, mlm_labels, t):

        # Put everything on device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mlm_labels = mlm_labels.to(device)
        t = t.to(device)

        # Obtain preditcitons from model
        model_outs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract MLM and CLS logits
        MLM_logits = model_outs["logits"]

        # Compute losses
        B, L, V = MLM_logits.shape
        MLM_loss = self.MLM_loss_fn(
            MLM_logits.view(B * L, V), 
            mlm_labels.view(B * L)
        ) # (B*L,)

        # Reshape to original shape
        MLM_loss = MLM_loss.view(B, L)

        # Calculate weights
        alpha_t = self.alpha(t)
        alpha_prime_t = self.alpha.dt(t)
        weights = alpha_prime_t / (1 - alpha_t) # (B, 1)

        # Multiply them together
        loss = (weights * MLM_loss).mean()

        # Print token level accuracy
        pred_tokens = MLM_logits.argmax(dim=-1)  # (B, L)
        mask = mlm_labels != -100
        accuracy = (pred_tokens[mask] == mlm_labels[mask]).float().mean()

        # Backprop
        self.step += 1
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update step
        self.optimizer.step()

        # Log into wandb
        if self.cfg.use_wandb:
            wandb.log({
                "MLM loss": loss.item(),
                "Total loss": loss.item()
            }, step=self.step)

        # Log into dicts
        self.train_MLM_loss[self.step] = loss.item()

        return loss, accuracy
    
    @torch.inference_mode
    def val_step(self, input_ids, attention_mask, mlm_labels, t):

        # Put everything on device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mlm_labels = mlm_labels.to(device)
        t = t.to(device)

        # Obtain predictions from model
        model_outs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract MLM and CLS logits
        MLM_logits = model_outs["logits"]

        # Compute losses
        B, L, V = MLM_logits.shape
        MLM_loss = self.MLM_loss_fn(
            MLM_logits.view(B * L, V), 
            mlm_labels.view(B * L)
        ) # (B * L, 1)

        # Reshape to original shape
        MLM_loss = MLM_loss.view(B, L)

        # Calculate weights
        alpha_t = self.alpha(t)
        alpha_prime_t = self.alpha.dt(t)
        weights = alpha_prime_t / (1 - alpha_t) # (B, 1)

        # Multiply them together
        loss = (weights * MLM_loss).mean()

        # Save token level accuracy
        pred_tokens = MLM_logits.argmax(dim=-1)  # (B, L)
        mask = mlm_labels != -100
        accuracy = (pred_tokens[mask] == mlm_labels[mask]).float().mean()

        # Log into wandb
        if self.cfg.use_wandb:
            wandb.log({
                "Val MLM loss": loss.item(),
                "Val token accuracy": accuracy.item(),
                "Val Total loss": loss.item()
            }, step=self.step)

        # Log into dicts
        self.test_MLM_loss[self.step] = loss.item()
        self.token_accuracy[self.step] = accuracy.item()

        return loss

    def train(self):

        # Set val loss to none
        best_val_loss = None

        # Define progress bar
        progress_bar = tqdm(
            range(self.cfg.num_epochs * len(self.train_dataloader())),
            total = len(self.train_dataloader()) * self.cfg.num_epochs,
            colour="GREEN"
        )

        for epoch in range(self.cfg.num_epochs):

            self.model.train()

            for batch in self.train_dataloader():
                l, acc = self.train_step(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["MLM_labels"],
                    batch["t"]
                )

                progress_bar.set_description(
                    f"Epoch: {epoch + 1}. T: {l:.4f}. Acc: {acc:.4f}"
                )
                progress_bar.update()

            # Scheduler step
            self.scheduler.step()
            self.current_epoch += 1

            # Start validation
            self.model.eval()

            val_losses = []
            for batch in self.test_dataloader():
                l = self.val_step(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["MLM_labels"],
                    batch["t"]
                )
                val_losses.append(l.item())
            
            avg_val_loss = np.mean(val_losses)
            # Save best model
            if best_val_loss is None or avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_model_and_metadata()

            
    def _save_model_and_metadata(self):

        os.makedirs(self.cfg.save_path, exist_ok=True)

        # Save model
        save_path = os.path.join(self.cfg.save_path, "weights.pt")
        torch.save(self.model.state_dict(), save_path)
        
        # Save metadata
        metadata_path = os.path.join(self.cfg.save_path, "metadata.json")
        self.cfg.save_cfg(metadata_path)

def save_loss_plot(
    train_MLM_losses: dict[int, float],
    test_MLM_losses: dict[int, float],
    token_accs: dict[int, float],
    path: str,
    num_params: float
):
    
    # Extract x axes
    x_train_ax = np.array(list(train_MLM_losses.keys()))
    x_val_ax = np.array(list(token_accs.keys()))

    # Convert everything to numpy
    train_MLM_loss = np.array(list(train_MLM_losses.values()))
    test_MLM_loss = np.array(list(test_MLM_losses.values()))
    token_acc = np.array(list(token_accs.values()))

    # Define figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # ---- consistent colors ----
    cmap = plt.cm.get_cmap("tab10")

    # Plot values
    plt.plot(x_train_ax, train_MLM_loss, color=cmap(1), label="Train MLM Loss")
    plt.plot(x_val_ax, test_MLM_loss, color=cmap(3), label="Test MLM Loss")
    plt.plot(x_val_ax, token_acc, color=cmap(4), label="Token Accuracy (val)")

    # Set title and axes labels
    plt.title(f"DiffusionDNABERT ({num_params}M)")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Loss")

    ax.spines[["right", "top"]].set_visible(False)

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(path, "loss.png"), format="png", dpi=300)
    plt.show()
    plt.close()

def main():

    cfg = TrainerNTDiffusionCFG()
    trainer = TrainerNTDiffusion(
        cfg,
    )

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer.train()
        
    if trainer.cfg.loss_path:

        # Ensure path exists
        os.makedirs(trainer.cfg.loss_path, exist_ok=True)

        # Calculate model size
        model_size = sum(p.numel() for p in trainer.model.parameters()) / 1e6
        save_loss_plot(
            trainer.train_MLM_loss, 
            trainer.test_MLM_loss, 
            trainer.token_accuracy,
            trainer.cfg.loss_path,
            model_size
        )

if __name__ == "__main__":
    main()


