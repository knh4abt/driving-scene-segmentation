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
    """Atomically save `state` to `path`, keeping the prior file as `<path>.prev`.

    Sequence:
      1. Write to `<path>.tmp` and fsync to flush bytes to disk.
      2. Rotate the current `<path>` to `<path>.prev` (skipped if first save).
      3. Atomically rename `<path>.tmp` -> `<path>`.

    `os.replace` is atomic on Windows and POSIX, so `<path>` itself is never
    partially written. A crash between steps 2 and 3 leaves `<path>.prev` and
    `<path>.tmp` on disk; `load_checkpoint` recovers from that state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    prev = path.with_suffix(path.suffix + ".prev")

    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())

    if path.exists():
        os.replace(path, prev)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    """Load `path`, falling back to `<path>.prev` if `<path>` is missing or corrupt.

    Recovers from a crash that hit between the two renames inside
    `save_checkpoint`. If a complete `<path>.tmp` is present, promote it so
    later loads see the newest good state.
    """
    path = Path(path)
    prev = path.with_suffix(path.suffix + ".prev")
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        return torch.load(path, map_location=map_location)
    except (FileNotFoundError, RuntimeError, EOFError) as e:
        if tmp.exists():
            try:
                state = torch.load(tmp, map_location=map_location)
                os.replace(tmp, path)
                print(f"[recovery] promoted {tmp.name} -> {path.name} after crash mid-save")
                return state
            except (RuntimeError, EOFError):
                pass
        if prev.exists():
            print(f"[recovery] loading {prev.name} ({path.name} unreadable: {e.__class__.__name__})")
            return torch.load(prev, map_location=map_location)
        raise


def poly_lr(base_lr: float, cur_iter: int, total_iters: int, power: float = 0.9,
            warmup_iters: int = 0) -> float:
    """Linear warmup for `warmup_iters`, then polynomial decay over the rest.

    `warmup_iters=0` (default) recovers pure poly decay, matching the recipe
    used by DeepLab/SGD. SegFormer/AdamW needs warmup to avoid the freshly
    initialized decode head collapsing into a degenerate dominant-class basin.
    """
    if warmup_iters > 0 and cur_iter < warmup_iters:
        return base_lr * (cur_iter + 1) / warmup_iters
    decay_iter = cur_iter - warmup_iters
    decay_total = max(total_iters - warmup_iters, 1)
    return base_lr * (1.0 - decay_iter / decay_total) ** power
