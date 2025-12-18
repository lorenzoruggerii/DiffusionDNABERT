import torch

from torch import nn
from transformers import AutoModelForMaskedLM
from config import DiffusionNTCFG

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species" # NT path

class DiffusionNT(nn.Module):

    def __init__(self, cfg: DiffusionNTCFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        self.head = nn.Linear(self.backbone.config.hidden_size, cfg.num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        NT_outs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Extract MLM_outputs and hidden states
        MLM_outputs = NT_outs["logits"]
        hidden_states = NT_outs["hidden_states"]

        # Compute CLS logits
        CLS_token = hidden_states[-1][:, 0, :]
        CLS_logits = self.head(CLS_token)

        return {
            "MLM_logits": MLM_outputs,
            "CLS_logits": CLS_logits
        }

