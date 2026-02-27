
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across:
    - Python
    - NumPy
    - PyTorch 
    """

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Seed set to {seed}")