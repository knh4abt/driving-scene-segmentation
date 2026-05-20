# Experiment Plan

## Setup (shared across both runs)

| Item              | Value                                                       |
|-------------------|-------------------------------------------------------------|
| Dataset           | Cityscapes, fine annotations only                           |
| Train / val split | 2,975 / 500 images (official)                               |
| Eval classes      | 19 (official trainId mapping; void -> ignore_index=255)     |
| Input crop        | 512 x 512 random crop                                       |
| Augmentation      | random scale [0.5, 2.0], horizontal flip                    |
| Normalization     | ImageNet mean/std                                           |
| Loss              | CrossEntropyLoss(ignore_index=255)                          |
| Optimizer         | SGD (DeepLab) / AdamW (SegFormer)                           |
| LR schedule       | Polynomial decay, power 0.9                                 |
| Epochs            | 200                                                         |
| Batch size        | 8                                                           |
| Precision         | AMP / fp16                                                  |
| Eval              | single-scale, full 1024 x 2048 resolution                   |
| Seed              | 42 (CLI: `--seed`)                                          |

## Run A: DeepLabV3+ / ResNet-50

- ImageNet-pretrained ResNet-50 backbone (torchvision).
- ASPP rates {6, 12, 18}, output stride 16.
- SGD, momentum 0.9, weight decay 1e-4, base LR 1e-2.
- Expected val mIoU: 78 to 80% (paper / public reproductions).

## Run B: SegFormer-B0

- HuggingFace `nvidia/mit-b0` pretrained encoder + SegFormer head.
- AdamW, weight decay 1e-2, base LR 6e-5, linear warmup 1500 iters.
- Expected val mIoU: 76 to 78% at roughly 10x fewer parameters.

## What to look for

- DeepLabV3+ is expected to be slightly higher in absolute mIoU but
  considerably heavier and slower per iteration.
- SegFormer-B0 is expected to be close in mIoU at a fraction of the
  parameter count, which is the relevant axis for embedded deployment.

## Error analysis

Classes that are typically hardest on Cityscapes, with reasons:

| Class pair                     | Why it's hard                                       |
|--------------------------------|-----------------------------------------------------|
| wall vs. fence                 | Both vertical static structures, ambiguous edges    |
| rider vs. motorcycle / bicycle | Rider (human) and vehicle touch at the boundary     |
| truck vs. bus                  | Similar large-vehicle silhouettes, both rare        |
| terrain vs. vegetation         | Both natural, low texture contrast at distance      |
| train                          | Very rare in val (~0.2% pixels), high variance      |

Planned outputs:

1. Per-class IoU table from `scripts/evaluate.py`.
2. Confusion matrix heatmap.
3. Failure-case gallery: a few images per model where IoU drops most, with
   ground truth and prediction overlaid.

## Compute budget (single 12 GB GPU)

| Model        | Time / epoch (approx) | Total (200 ep) | Peak VRAM |
|--------------|----------------------:|---------------:|----------:|
| DeepLabV3+   | 6 to 8 min            | ~24 h          | ~9 GB     |
| SegFormer-B0 | 4 to 5 min            | ~16 h          | ~7 GB     |
