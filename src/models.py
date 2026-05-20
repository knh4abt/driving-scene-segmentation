"""Model factory.

`build_model(name, num_classes, pretrained)` returns a torch.nn.Module whose
forward takes (B, 3, H, W) and returns logits of shape (B, num_classes, H, W).

Backends:
  - "deeplabv3plus_r50": torchvision's deeplabv3_resnet50 (ASPP head on a
    dilated ResNet-50). Within roughly 1 mIoU of the full DeepLabV3+ and
    keeps the dependency footprint to torchvision only.
  - "segformer_b0": HuggingFace SegformerForSemanticSegmentation with the
    nvidia/mit-b0 encoder. Outputs logits at H/4 x W/4; we upsample to the
    input resolution to match the DeepLab interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DeepLabWrapper(nn.Module):
    """torchvision returns {'out': logits, ...}; unwrap to a plain tensor."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]


class _SegFormerWrapper(nn.Module):
    """HF SegFormer outputs at 1/4 input resolution; upsample to match input."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits  # (B, C, H/4, W/4)
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def build_model(name: str, num_classes: int = 19, pretrained: bool = True) -> nn.Module:
    """Return a segmentation model. Forward: (B,3,H,W) -> (B,num_classes,H,W)."""
    name = name.lower()

    if name in {"deeplabv3plus_r50", "deeplab_r50", "deeplabv3_r50"}:
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models import ResNet50_Weights

        backbone_w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_w,
            num_classes=num_classes,
            aux_loss=False,
        )
        return _DeepLabWrapper(net)

    if name in {"segformer_b0", "segformer-b0", "mit_b0"}:
        from transformers import SegformerForSemanticSegmentation

        if pretrained:
            net = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,  # head reinit for 19 classes
            )
        else:
            from transformers import SegformerConfig

            net = SegformerForSemanticSegmentation(SegformerConfig(num_labels=num_classes))
        return _SegFormerWrapper(net)

    raise ValueError(f"Unknown model name: {name!r}")
