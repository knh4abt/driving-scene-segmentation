"""Cityscapes dataset.

Cityscapes label PNGs use 34 raw class IDs (0..33). The official benchmark
evaluates only 19 classes; the rest must be remapped to ignore_index=255 so
the loss skips them. The mapping below comes from the official labels.py
in cityscapesScripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 34 -> 19 trainId mapping. Index = raw labelId, value = trainId. 255 = ignore.
# fmt: off
_ID_TO_TRAINID = np.array([
    255, 255, 255, 255, 255, 255, 255,   # 0..6
      0,   1, 255, 255,                  # 7..10  road, sidewalk
      2,   3,   4, 255, 255, 255,        # 11..16 building, wall, fence
      5, 255,   6,   7,                  # 17..20 pole, traffic light, traffic sign
      8,   9,  10,                       # 21..23 vegetation, terrain, sky
     11,  12,                            # 24..25 person, rider
     13,  14,  15, 255, 255,             # 26..30 car, truck, bus
     16,  17,  18,                       # 31..33 train, motorcycle, bicycle
], dtype=np.uint8)
# fmt: on

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
NUM_CLASSES = 19
IGNORE_INDEX = 255


def encode_label(raw_label: np.ndarray) -> np.ndarray:
    """Map raw 34-class labelIds to 19-class trainIds (or 255)."""
    return _ID_TO_TRAINID[raw_label]


class Cityscapes(Dataset):
    """Cityscapes loader for the fine annotation set.

    Expected layout under `root`:
        leftImg8bit/{split}/<city>/<file>_leftImg8bit.png
        gtFine/{split}/<city>/<file>_gtFine_labelIds.png
    """

    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        img_dir = self.root / "leftImg8bit" / split
        lbl_dir = self.root / "gtFine" / split
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Cityscapes split not found: {img_dir}. "
                "Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip "
                "and extract them under `root`."
            )

        self.samples: list[tuple[Path, Path]] = []
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = img_path.name.replace("_leftImg8bit.png", "")
                lbl_path = lbl_dir / city_dir.name / f"{stem}_gtFine_labelIds.png"
                if lbl_path.exists():
                    self.samples.append((img_path, lbl_path))

        if not self.samples:
            raise RuntimeError(f"No samples found under {img_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        raw_label = np.array(Image.open(lbl_path), dtype=np.uint8)
        label = Image.fromarray(encode_label(raw_label))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
