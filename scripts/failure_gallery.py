"""Save the N val images with the lowest per-image mIoU as triptychs.

Per-image mIoU is computed only over classes actually present in that image's
ground truth (NaNs skipped), so an image is not penalized just for missing
rare classes. Output filenames embed rank, val index, and mIoU for easy
sorting and citation.

Usage:
    python scripts/failure_gallery.py --config configs/train.yaml \\
        --checkpoint runs/deeplabv3plus_r50/best.pt \\
        --num-samples 8 --out-dir docs/results/failures_deeplab
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import IGNORE_INDEX, NUM_CLASSES, Cityscapes
from src.metrics import ConfusionMatrixMeter
from src.models import build_model
from src.transforms import EvalTransform, IMAGENET_MEAN, IMAGENET_STD
from src.utils import colorize, load_checkpoint


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_t.cpu() * std + mean).clamp(0, 1).numpy()
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def per_image_miou(pred: np.ndarray, target: np.ndarray) -> float:
    meter = ConfusionMatrixMeter(NUM_CLASSES, IGNORE_INDEX)
    meter.update(torch.from_numpy(pred[None]), torch.from_numpy(target[None]))
    return meter.compute()["miou"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--out-dir", type=str, required=True)
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
    print(f"[failure] checkpoint={args.checkpoint}  model={model_name}")

    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_set = Cityscapes(cfg["data"]["root"], split="val", transform=EvalTransform())

    print(f"[failure] scoring {len(val_set)} val images...")
    scores = []
    with torch.no_grad():
        for i in tqdm(range(len(val_set))):
            image, target = val_set[i]
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
            miou = per_image_miou(pred, target.numpy().astype(np.int64))
            scores.append((miou, i))

    scores.sort()
    worst = scores[: args.num_samples]

    with torch.no_grad():
        for rank, (miou, idx) in enumerate(worst):
            image, target = val_set[int(idx)]
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            img_rgb = denormalize(image)
            gt_rgb = colorize(target.numpy().astype(np.uint8), IGNORE_INDEX)
            pr_rgb = colorize(pred, IGNORE_INDEX)
            triptych = np.concatenate([img_rgb, gt_rgb, pr_rgb], axis=1)
            Image.fromarray(triptych).save(
                out_dir / f"rank{rank+1:02d}_val{idx:04d}_miou{miou*100:05.2f}.png"
            )

    print(f"[failure] saved {len(worst)} worst-case images to {out_dir}")
    print("  worst per-image mIoUs:", [f"{m*100:.1f}" for m, _ in worst])


if __name__ == "__main__":
    main()
