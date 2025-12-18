from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import torch

@dataclass
class TrainerNTDiffusionCFG:
    
    # Hyperparameters
    lr: int = 0.00001
    num_epochs: int = 20
    batch_size: int = 16
    comb_factor: float = 4 # this should be reparametrized for having losses on the same scale
    seed: int = 42

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "DiffusionBERT"
    wandb_name: str = "DiffusionBERT"
    
    # Training Arguments
    dataset_path: str = "leannmlindsey/GUE"
    col_name: str = "enhancer"
    save_path: str = "models/Enhancers_ft_paper" # dir
    loss_path: Optional[str] = f"{save_path}/images"
    train_chroms: list[str] =  field(default_factory=lambda: ["chr1", "chr3", "chr4", "chr5"])
    test_chroms: list[str] = field(default_factory=lambda: ["chr2"])
    max_len: Optional[int] = None # the default is 2048 (do max(len(sequences)) / 6)
    reinit: bool = False

    def save_cfg(self, path: str):

        data = asdict(self)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class DiffusionNTCFG:
    # Parameters
    num_classes: int = 2

# Time step scheduler
class LinearAlpha:
    def __init__(self):
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)
        )
        
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)
    


    