"""Save side-by-side image | ground truth | prediction PNGs.

Usage:
    python scripts/visualize.py --config configs/train.yaml \
        --checkpoint runs/deeplabv3plus_r50/best.pt \
        --num-samples 8 --out-dir runs/deeplabv3plus_r50/vis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import Cityscapes, IGNORE_INDEX, NUM_CLASSES
from src.models import build_model
from src.transforms import EvalTransform, IMAGENET_MEAN, IMAGENET_STD
from src.utils import colorize, load_checkpoint


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    """Inverse of ImageNet normalization, returns uint8 (H, W, 3)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_t.cpu() * std + mean).clamp(0, 1).numpy()
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", cfg["model"]["name"])
    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_set = Cityscapes(cfg["data"]["root"], split="val", transform=EvalTransform())

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(val_set), size=min(args.num_samples, len(val_set)), replace=False)

    with torch.no_grad():
        for i in idxs:
            image, target = val_set[int(i)]
            image_b = image.unsqueeze(0).to(device)
            pred = model(image_b).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            img_rgb = denormalize(image)
            gt_rgb = colorize(target.numpy().astype(np.uint8), IGNORE_INDEX)
            pr_rgb = colorize(pred, IGNORE_INDEX)

            triptych = np.concatenate([img_rgb, gt_rgb, pr_rgb], axis=1)
            Image.fromarray(triptych).save(out_dir / f"sample_{int(i):04d}.png")

    print(f"[vis] saved {len(idxs)} images to {out_dir}")


if __name__ == "__main__":
    main()
