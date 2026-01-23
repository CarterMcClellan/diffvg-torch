"""
I/O utilities for diffvg_torch.
"""

import torch
import numpy as np
from PIL import Image


def get_device():
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def imwrite(image: torch.Tensor, path: str, gamma: float = 1.0):
    """Save image tensor to file with optional gamma correction."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if gamma != 1.0:
        image = np.power(np.clip(image, 0, 1), 1.0 / gamma)
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    Image.fromarray(image).save(path)
