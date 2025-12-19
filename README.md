# DNA Diffusion-BERT

This repository implements a **Discrete Diffusion Process** for DNA sequence generation and fine-tuning. It utilizes state-of-the-art genomic models like **DNABERT-2** and **Nucleotide Transformer**, reparameterizing Masked Language Modeling (MLM) as a diffusion process with iterative refinement.

I've made a blog post about generating human enhancers, you can read it [here](https://lorenzoruggerii.github.io/blog/2025/DiffusionDNABERT-new/).

---

## 📂 Project Structure

```text
└── src
    ├── config.py         # Configuration dataclasses & Diffusion schedulers
    ├── dataset.py        # Parallelized genomic data extraction (bigBed/FASTA)
    ├── generate.py       # Iterative refinement & confidence-based generation
    ├── model.py          # Model wrapper for Nucleotide Transformer
    ├── trainer_mlm.py    # Main Discrete Diffusion training loop
    └── trainer.py        # Base training utilities
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lorenzoruggerii/DiffusionDNABERT.git
   cd DiffusionDNABERT
   ```

2. **Install dependencies**: Ensure you have Python 3.9+ and install the necessary biological and machine learning libraries:
    ```bash
    pip install torch transformers datasets pyBigWig pyfaidx tqdm wandb
    ```

## 🧬 Data Pipeline

The `src/dataset.py` script handles the extraction of DNA sequences from genomic tracks. It performs a parallelized search across `bigBed` files and fetches the corresponding nucleotide sequences from a reference `FASTA` file.

**Example Command:**
```bash
python src/dataset.py \
    --bb_folder ./data/tracks \
    --chroms 1-22 \
    --fasta ./data/hg38.fa \
    --outpath ./data/processed_dataset \
    --num_peaks 1000
```

## 🚀 Training Logic

The training process uses a **Discrete Diffusion** objective. We define a linear scheduler $\alpha(t) = 1 - t$ to control the masking rate. 



To ensure the gradients are scaled correctly across different timesteps $t$, the MLM loss is weighted by:

$$\text{Weight}(t) = \frac{\alpha'(t)}{1 - \alpha(t)}$$

**Run Training:**
```bash
python src/trainer_mlm.py
```

**Note**: Configuration settings such as `lr`, `batch_size`, and `use_wandb` can be adjusted in `src/config.py`.

## 🔍 Sequence Generation & Refinement

The generation script in `src/generate.py` utilizes **Confidence-based Iterative Refinement**. Unlike standard one-pass decoding, this method ensures high-quality motifs by:

1.  **Initial Generation:** Unmasking tokens with the highest prediction confidence.
2.  **Refinement Pass:** Identifying low-confidence tokens (below a threshold), re-masking them, and regenerating them in the context of the new sequence.



**Generate Sequences:**
```bash
python src/generate.py \
    --model_path ./models/weights.pt \
    --num_sequences 50 \
    --length 200 \
    --outpath ./results/generated.fasta \
    --top_p 0.9
```

## ⚙️ Configuration Reference

The `TrainerNTDiffusionCFG` dataclass in `src/config.py` manages all hyperparameters:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `lr` | `0.00001` | Learning rate for AdamW |
| `num_epochs` | `20` | Total training epochs |
| `comb_factor` | `4.0` | Scaling for loss reparameterization |
| `use_wandb` | `False` | Toggle Weights & Biases logging |
| `max_len` | `2048` | Maximum sequence length for the model |