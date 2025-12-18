import torch
import torch.nn.functional as F
import argparse
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
DNABERT = "zhihan1996/DNABERT-2-117M"

def generate_dna_sequence(
    model, 
    tokenizer, 
    length=100,
    strategy="confidence",  # "confidence", "random", "left-to-right"
    temperature=1.0,
    top_p=0.9,
    num_unmask_per_step=20,
    initial_sequence=None,
    device="cuda",
    # New refinement parameters
    min_confidence=0.7,  # Minimum confidence threshold for satisfaction
    max_refinement_iterations=5,  # Maximum number of refinement passes
    refinement_percentage=1,  # Fraction of low-confidence tokens to remask per iteration
):
    """
    Generate DNA sequences using a DNABERT masked language model with iterative refinement.
    
    Args:
        strategy: 
            - "confidence": Unmask tokens with highest prediction confidence
            - "random": Randomly unmask tokens
            - "left-to-right": Unmask sequentially
        temperature: Sampling temperature (lower = more conservative)
        top_p: Nucleus sampling threshold
        min_confidence: Tokens below this confidence will be refined
        max_refinement_iterations: Maximum number of refinement passes
        refinement_percentage: Fraction of low-confidence tokens to remask per iteration
    """
    model.eval()
    
    # Initialize with mask tokens
    if initial_sequence is None:
        input_ids = torch.full(
            (1, length), 
            tokenizer.mask_token_id, 
            dtype=torch.long,
            device=device
        )
    else:
        tok_out = tokenizer(initial_sequence, return_tensors="pt", padding="max_length", max_length=length)
        input_ids = tok_out["input_ids"].to(device)
        # Replace non-special tokens with masks
        special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
        mask = ~torch.isin(input_ids, torch.tensor(list(special_tokens), device=device))
        input_ids[mask] = tokenizer.mask_token_id
    
    attention_mask = torch.ones_like(input_ids)
    
    # Track confidence scores for each position
    token_confidences = torch.zeros(length, device=device)
    
    # Phase 1: Initial generation
    step = 0
    while (input_ids == tokenizer.mask_token_id).any():
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Get masked positions
            masked_positions = (input_ids == tokenizer.mask_token_id)
            
            if strategy == "confidence":
                max_probs, predictions = probs.max(dim=-1)
                confidence_scores = max_probs.clone()
                confidence_scores[~masked_positions] = -float('inf')
                
                num_to_unmask = min(num_unmask_per_step, masked_positions.sum().item())
                _, positions_to_unmask = torch.topk(
                    confidence_scores.squeeze(0), 
                    k=num_to_unmask
                )
                
            elif strategy == "random":
                masked_indices = masked_positions.squeeze(0).nonzero(as_tuple=True)[0]
                num_to_unmask = min(num_unmask_per_step, len(masked_indices))
                perm = torch.randperm(len(masked_indices), device=device)
                positions_to_unmask = masked_indices[perm[:num_to_unmask]]
                
            elif strategy == "left-to-right":
                masked_indices = masked_positions.squeeze(0).nonzero(as_tuple=True)[0]
                num_to_unmask = min(num_unmask_per_step, len(masked_indices))
                positions_to_unmask = masked_indices[:num_to_unmask]
            
            # Sample tokens for selected positions and store their confidence
            for pos in positions_to_unmask:
                pos_probs = probs[0, pos]
                
                # Optional: nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(pos_probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    pos_probs[indices_to_remove] = 0
                    pos_probs = pos_probs / pos_probs.sum()
                
                # Sample from the distribution
                sampled_token = torch.multinomial(pos_probs, num_samples=1)
                input_ids[0, pos] = sampled_token
                
                # Store the confidence for this token
                token_confidences[pos] = pos_probs[sampled_token].item()
        
        step += 1
        if step % 10 == 0:
            num_masked = (input_ids == tokenizer.mask_token_id).sum().item()
    
    for refinement_iter in range(max_refinement_iterations):
        # Recalculate confidence for all tokens with current context
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits / temperature, dim=-1)
            
            # Get confidence for each current token
            current_tokens = input_ids[0]
            for i in range(length):
                if current_tokens[i] not in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}:
                    token_confidences[i] = probs[0, i, current_tokens[i]].item()
        
        # Find low-confidence tokens
        special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
        non_special_mask = ~torch.isin(current_tokens, torch.tensor(list(special_tokens), device=device))
        
        low_confidence_positions = (token_confidences < min_confidence) & non_special_mask
        num_low_confidence = low_confidence_positions.sum().item()
        
        avg_confidence = token_confidences[non_special_mask].mean().item()
        min_conf = token_confidences[non_special_mask].min().item()
        
        if num_low_confidence == 0:
            break
        
        # Select tokens to remask (worst performing ones)
        num_to_remask = max(1, int(num_low_confidence * refinement_percentage))
        low_conf_scores = token_confidences.clone()
        low_conf_scores[~low_confidence_positions] = float('inf')
        
        _, positions_to_remask = torch.topk(
            low_conf_scores,
            k=num_to_remask,
            largest=False
        )
        
        # Remask selected positions
        input_ids[0, positions_to_remask] = tokenizer.mask_token_id
        
        # Regenerate masked tokens
        while (input_ids == tokenizer.mask_token_id).any():
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits / temperature
                probs = F.softmax(logits, dim=-1)
                
                masked_positions = (input_ids == tokenizer.mask_token_id)
                
                # Use confidence-based unmasking for refinement
                max_probs, _ = probs.max(dim=-1)
                confidence_scores = max_probs.clone()
                confidence_scores[~masked_positions] = -float('inf')
                
                num_to_unmask = min(num_unmask_per_step, masked_positions.sum().item())
                _, positions_to_unmask = torch.topk(
                    confidence_scores.squeeze(0),
                    k=num_to_unmask
                )
                
                for pos in positions_to_unmask:
                    pos_probs = probs[0, pos]
                    
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(pos_probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = False
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        pos_probs[indices_to_remove] = 0
                        pos_probs = pos_probs / pos_probs.sum()
                    
                    sampled_token = torch.multinomial(pos_probs, num_samples=1)
                    input_ids[0, pos] = sampled_token
                    token_confidences[pos] = pos_probs[sampled_token].item()
    
    # Final statistics
    final_avg_conf = token_confidences[non_special_mask].mean().item()
    final_min_conf = token_confidences[non_special_mask].min().item()
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", type=str, required=True, help="Where to find the model.")
    p.add_argument("--num_sequences", type=int, required=True, help="Num sequences to generate")
    p.add_argument("--outpath", required=True, type=str, help="Where to save sequences.")
    p.add_argument("--length", required=True, type=int, help="Number of generated tokens")
    p.add_argument("--top_p", type=float, required=False, default=0.9, help="Top-p used from the generation.")
    

    args = p.parse_args()

    # Create directory
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    model = AutoModelForMaskedLM.from_pretrained(
        DNABERT,
        trust_remote_code=True
    ).to(device)

    # Load your fine-tuned state dict
    state_dict = torch.load(args.model_path, map_location=device)

    # Load weights into the model
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(
        DNABERT,
        trust_remote_code=True
    )

    sequences = []
    for i in tqdm(range(args.num_sequences), colour="GREEN", desc="Generating..."):
        sequence = generate_dna_sequence(
            model,
            tokenizer,
            args.length,
            top_p=args.top_p,
            strategy="confidence",
            device=device,
            min_confidence=0 # so that no replacement will occur
        )
        sequences.append(sequence.replace(" ", ""))

    with open(args.outpath, "w") as fopen:
        for i, s in enumerate(sequences):
            fopen.write(f">Seq_{i}\n{s}\n")

    print(f"Sequences saved in {args.outpath}.")

if __name__ == "__main__":
    main()