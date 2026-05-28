"""Save a confusion matrix heatmap for a checkpoint on the Cityscapes val split.

Rows are ground-truth class, columns are predicted class. Each row is normalized
so the diagonal reads "recall for this class" and off-diagonal cells read
"fraction of this class's pixels predicted as that other class."

Usage:
    python scripts/analyze_confusion.py --config configs/train.yaml \\
        --checkpoint runs/deeplabv3plus_r50/best.pt \\
        --out-png docs/results/confusion_deeplab.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, Cityscapes
from src.metrics import ConfusionMatrixMeter
from src.models import build_model
from src.transforms import EvalTransform
from src.utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-png", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", cfg["model"]["name"])
    print(f"[confusion] checkpoint={args.checkpoint}  model={model_name}")

    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_set = Cityscapes(cfg["data"]["root"], split="val", transform=EvalTransform())
    val_loader = DataLoader(
        val_set, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    meter = ConfusionMatrixMeter(NUM_CLASSES, IGNORE_INDEX)
    with torch.no_grad():
        for image, target in tqdm(val_loader, desc="confusion"):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(image).argmax(dim=1)
            meter.update(pred, target)

    M = meter.mat.astype(np.float64)
    row_sums = M.sum(axis=1, keepdims=True)
    Mn = np.where(row_sums > 0, M / np.maximum(row_sums, 1), 0.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(Mn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(f"Confusion matrix (row-normalized, %) -- {model_name}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = Mn[i, j]
            if v >= 0.01:
                color = "white" if v < 0.5 else "black"
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of GT pixels")
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"[confusion] saved {out_png}")


if __name__ == "__main__":
    main()
