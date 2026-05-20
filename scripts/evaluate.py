"""Evaluate a checkpoint on the Cityscapes val split.

Usage:
    python scripts/evaluate.py --config configs/train.yaml \
        --checkpoint runs/deeplabv3plus_r50/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    p.add_argument("--out-json", type=str, default=None,
                   help="If given, write per-class IoU and mIoU to this JSON file.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    # Use the model name stored in the checkpoint when available; the YAML
    # may have changed since training.
    model_name = ckpt.get("model_name", cfg["model"]["name"])
    print(f"[eval] checkpoint={args.checkpoint}  model={model_name}")

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
        for image, target in tqdm(val_loader, desc="val"):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(image).argmax(dim=1)
            meter.update(pred, target)

    res = meter.compute()
    iou = res["iou_per_class"]

    print()
    print(f"{'class':<15} {'IoU (%)':>8}")
    print("-" * 25)
    for name, v in zip(CLASS_NAMES, iou):
        v_str = f"{v*100:7.2f}" if v == v else "    nan"
        print(f"{name:<15} {v_str}")
    print("-" * 25)
    print(f"{'mIoU':<15} {res['miou']*100:7.2f}")
    print(f"{'pixel_acc':<15} {res['pixel_acc']*100:7.2f}")

    if args.out_json:
        # Float-cast NaNs so the file stays valid JSON.
        per_class = {
            name: (float(v) if v == v else None)
            for name, v in zip(CLASS_NAMES, iou)
        }
        payload = {
            "checkpoint": str(args.checkpoint),
            "model_name": model_name,
            "miou": float(res["miou"]),
            "pixel_acc": float(res["pixel_acc"]),
            "iou_per_class": per_class,
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[eval] wrote results to {out_path}")


if __name__ == "__main__":
    main()
