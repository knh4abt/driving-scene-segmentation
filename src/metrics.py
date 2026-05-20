"""Streaming confusion-matrix-based mIoU.

Per-class IoU = TP / (TP + FP + FN). Computing this from a confusion matrix
accumulated over the whole eval set is the standard approach in mmseg and
the official Cityscapes scripts. Per-batch averaging is wrong when rare
classes are missing from some batches.

Workflow:
  1. Accumulate a (C, C) matrix M with M[gt, pred] = pixel count.
  2. After eval:
        TP_c = M[c, c]
        FP_c = M[:, c].sum() - TP_c
        FN_c = M[c, :].sum() - TP_c
        IoU_c = TP_c / (TP_c + FP_c + FN_c)
        mIoU  = mean over classes (NaN for classes with no pixels)
"""

from __future__ import annotations

import numpy as np
import torch


class ConfusionMatrixMeter:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: (B, H, W) int64. target may contain ignore_index."""
        pred = pred.detach().cpu().numpy().reshape(-1)
        target = target.detach().cpu().numpy().reshape(-1)

        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]

        # Encode (gt, pred) as one int = gt * C + pred, then bincount.
        k = self.num_classes
        idx = target.astype(np.int64) * k + pred.astype(np.int64)
        self.mat += np.bincount(idx, minlength=k * k).reshape(k, k)

    def compute(self) -> dict:
        m = self.mat.astype(np.float64)
        tp = np.diag(m)
        fp = m.sum(axis=0) - tp
        fn = m.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        miou = float(np.nanmean(iou))
        pixel_acc = float(tp.sum() / max(m.sum(), 1))
        return {"iou_per_class": iou, "miou": miou, "pixel_acc": pixel_acc}

    def reset(self) -> None:
        self.mat[:] = 0
