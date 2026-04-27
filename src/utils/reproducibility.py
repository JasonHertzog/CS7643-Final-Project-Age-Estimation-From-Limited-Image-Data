import torch
import numpy as np
import random

def set_seed(seed=42):
    """
    Sets all random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check if CUDA (GPU) is available before seeding it
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Log the seed instead of a simple print
    print(f"Reproducibility: Random seed set to {seed}")