"""Color palette, seeding, checkpoints, polynomial LR."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

# Cityscapes colors per trainId, in trainId order. Source: cityscapesScripts.
CITYSCAPES_PALETTE = np.array(
    [
        (128,  64, 128),  # road
        (244,  35, 232),  # sidewalk
        ( 70,  70,  70),  # building
        (102, 102, 156),  # wall
        (190, 153, 153),  # fence
        (153, 153, 153),  # pole
        (250, 170,  30),  # traffic light
        (220, 220,   0),  # traffic sign
        (107, 142,  35),  # vegetation
        (152, 251, 152),  # terrain
        ( 70, 130, 180),  # sky
        (220,  20,  60),  # person
        (255,   0,   0),  # rider
        (  0,   0, 142),  # car
        (  0,   0,  70),  # truck
        (  0,  60, 100),  # bus
        (  0,  80, 100),  # train
        (  0,   0, 230),  # motorcycle
        (119,  11,  32),  # bicycle
    ],
    dtype=np.uint8,
)


def colorize(label: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """(H, W) int label -> (H, W, 3) uint8 RGB. Ignored pixels become black."""
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    valid = (label != ignore_index) & (label < len(CITYSCAPES_PALETTE))
    out[valid] = CITYSCAPES_PALETTE[label[valid]]
    return out


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs. cuDNN nondeterminism is left on."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str | os.PathLike, state: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)


def poly_lr(base_lr: float, cur_iter: int, total_iters: int, power: float = 0.9) -> float:
    """lr(t) = base_lr * (1 - t / total)^power."""
    return base_lr * (1.0 - cur_iter / max(total_iters, 1)) ** power
