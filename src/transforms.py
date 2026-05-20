"""Joint image/mask augmentations.

The same random crop, flip and scale must be applied to both the image and
the label, but with different interpolation: bilinear for the image, nearest
for the label (averaging integer class IDs would produce invalid labels).
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TrainTransform:
    """Random scale -> random crop -> horizontal flip -> normalize."""

    def __init__(self, crop_size: int, scale_range=(0.5, 2.0), ignore_index: int = 255):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.ignore_index = ignore_index

    def __call__(self, image: Image.Image, label: Image.Image):
        # Random rescale.
        scale = random.uniform(*self.scale_range)
        w, h = image.size
        nw, nh = int(w * scale), int(h * scale)
        image = image.resize((nw, nh), Image.BILINEAR)
        label = label.resize((nw, nh), Image.NEAREST)

        # Pad if smaller than the crop. Label pad value = ignore_index so the
        # loss skips the padded region.
        pad_w = max(self.crop_size - nw, 0)
        pad_h = max(self.crop_size - nh, 0)
        if pad_w > 0 or pad_h > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            label = TF.pad(label, [0, 0, pad_w, pad_h], fill=self.ignore_index)
        nw, nh = image.size

        # Random crop.
        x = random.randint(0, nw - self.crop_size)
        y = random.randint(0, nh - self.crop_size)
        image = TF.crop(image, y, x, self.crop_size, self.crop_size)
        label = TF.crop(label, y, x, self.crop_size, self.crop_size)

        # Horizontal flip.
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # To tensor + normalize image; label is int64 class indices.
        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        label = torch.as_tensor(np.array(label), dtype=torch.long)
        return image, label


class EvalTransform:
    """Full-resolution eval. No crop, no flip."""

    def __call__(self, image: Image.Image, label: Image.Image):
        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        label = torch.as_tensor(np.array(label), dtype=torch.long)
        return image, label
