"""Train a segmentation model on Cityscapes (single GPU, AMP).

Examples:
    python scripts/train.py --config configs/train.yaml --seed 42
    python scripts/train.py --config configs/train.yaml --model segformer_b0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import IGNORE_INDEX, NUM_CLASSES, Cityscapes
from src.metrics import ConfusionMatrixMeter
from src.models import build_model
from src.transforms import EvalTransform, TrainTransform
from src.utils import poly_lr, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--model", type=str, default=None,
                   help="Override model.name in the config (e.g. segformer_b0).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Resume from runs/<experiment_name>/last.pt if it exists.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_optimizer(model: nn.Module, model_name: str, optim_cfg: dict):
    """SGD for DeepLab, AdamW for SegFormer."""
    if "deeplab" in model_name:
        c = optim_cfg["deeplab"]
        opt = torch.optim.SGD(
            model.parameters(),
            lr=c["lr"], momentum=c["momentum"], weight_decay=c["weight_decay"],
        )
        return opt, c["lr"]
    if "segformer" in model_name or "mit" in model_name:
        c = optim_cfg["segformer"]
        opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        return opt, c["lr"]
    raise ValueError(f"No optimizer recipe for model {model_name!r}")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    meter = ConfusionMatrixMeter(NUM_CLASSES, IGNORE_INDEX)
    for image, target in tqdm(loader, desc="val", leave=False):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(image).argmax(dim=1)
        meter.update(pred, target)
    return meter.compute()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg["model"]["name"] = args.model
        cfg["experiment_name"] = args.model
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} model={cfg['model']['name']} seed={args.seed}")

    # Data.
    train_tf = TrainTransform(
        crop_size=cfg["data"]["crop_size"],
        scale_range=tuple(cfg["data"]["scale_range"]),
        ignore_index=cfg["data"]["ignore_index"],
    )
    train_set = Cityscapes(cfg["data"]["root"], split="train", transform=train_tf)
    val_set = Cityscapes(cfg["data"]["root"], split="val", transform=EvalTransform())
    print(f"[data] train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    # Model, optimizer, loss.
    model = build_model(
        cfg["model"]["name"], num_classes=NUM_CLASSES,
        pretrained=cfg["model"]["pretrained_backbone"],
    ).to(device)
    optimizer, base_lr = build_optimizer(model, cfg["model"]["name"], cfg["optim"])
    poly_power = cfg["optim"]["poly_power"]
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["data"]["ignore_index"])

    epochs = cfg["train"]["epochs"]
    iters_per_epoch = len(train_loader)
    total_iters = epochs * iters_per_epoch

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(cfg["log"]["out_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    start_epoch = 0
    global_iter = 0

    # Optionally resume from last.pt. Restores model, optimizer, AMP scaler,
    # epoch counter, global iter and best mIoU so training continues seamlessly.
    last_ckpt = out_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"]
        global_iter = ckpt["global_iter"]
        best_miou = ckpt["best_miou"]
        print(f"[resume] from {last_ckpt} at epoch {start_epoch}, best mIoU so far = {best_miou*100:.2f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        for image, target in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            lr = poly_lr(base_lr, global_iter, total_iters, poly_power)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(image)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_iter += 1

        results = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"[epoch {epoch+1:3d}] lr={lr:.2e} "
            f"val_mIoU={results['miou']*100:.2f} "
            f"pixel_acc={results['pixel_acc']*100:.2f} "
            f"time={elapsed:.1f}s"
        )

        if results["miou"] > best_miou:
            best_miou = results["miou"]
            save_checkpoint(out_dir / "best.pt", {
                "epoch": epoch + 1,
                "model_name": cfg["model"]["name"],
                "model_state": model.state_dict(),
                "miou": results["miou"],
                "iou_per_class": results["iou_per_class"].tolist(),
                "config": cfg,
            })
            print(f"           new best, saved to {out_dir/'best.pt'}")

        # Always overwrite last.pt so we can resume from the most recent epoch.
        save_checkpoint(out_dir / "last.pt", {
            "epoch": epoch + 1,
            "global_iter": global_iter,
            "best_miou": best_miou,
            "model_name": cfg["model"]["name"],
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "miou": results["miou"],
            "config": cfg,
        })

    print(f"[done] best val mIoU = {best_miou*100:.2f}")


if __name__ == "__main__":
    main()
