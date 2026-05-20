# Driving Scene Segmentation (Cityscapes)

Semantic segmentation on the Cityscapes dataset. Two models are trained and
compared on the official 19-class benchmark:

1. DeepLabV3+ with a ResNet-50 backbone
2. SegFormer-B0

Both are evaluated by mean IoU overall and per class on the validation split.

## Task

For every pixel of a 1024x2048 street image, predict one of 19 classes:
road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle,
bicycle.

The metric is mean Intersection over Union:

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \qquad \text{mIoU} = \frac{1}{19}\sum_{c=1}^{19} \text{IoU}_c$$

Pixel accuracy is not reported as the main metric because road alone covers
roughly 35% of pixels in a typical scene.

## Repo layout

```
configs/        train.yaml
docs/           experiment_plan.md
src/
  dataset.py    Cityscapes loader + 34->19 trainId remap
  transforms.py joint image/mask augmentations
  models.py     build_model("deeplabv3plus_r50" | "segformer_b0")
  metrics.py    confusion-matrix-based mIoU
  utils.py      color palette, seeding, checkpoint helpers, poly LR
scripts/
  train.py
  evaluate.py
  visualize.py
requirements.txt
```

## Dataset setup

Cityscapes requires registration.

1. Create an account at https://www.cityscapes-dataset.com/.
2. Download from the Downloads page:
   - `leftImg8bit_trainvaltest.zip` (about 11 GB) - RGB images
   - `gtFine_trainvaltest.zip` (about 241 MB) - fine annotations
3. Extract both into one folder, e.g. `D:/datasets/cityscapes`:

```
cityscapes/
  leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
  gtFine/{train,val,test}/<city>/*_gtFine_labelIds.png
```

4. Set `data.root` in `configs/train.yaml` to that folder.

Only the fine annotations are used. Published numbers for DeepLabV3+ and
SegFormer are reported on the fine set, so this matches the literature.

## Setup

```bash
conda create -n dss python=3.10 -y
conda activate dss
pip install -r requirements.txt
```

## Hardware

The shipped config (`configs/train.yaml`) targets a single 6 GB consumer
GPU (tested on an NVIDIA GTX 1660 Ti): crop 384, batch size 4, 100 epochs,
mixed precision. On a 12 GB+ card you can raise the crop to 512 and the
batch size to 8 and roughly halve the wall-clock per run.

## Train

```bash
# DeepLabV3+ / ResNet-50 (default)
python scripts/train.py --config configs/train.yaml --seed 42

# SegFormer-B0
python scripts/train.py --config configs/train.yaml --model segformer_b0 --seed 42

# Resume an interrupted run from runs/<experiment>/last.pt
python scripts/train.py --config configs/train.yaml --resume
```

## Evaluate

```bash
python scripts/evaluate.py --config configs/train.yaml \
    --checkpoint runs/deeplabv3plus_r50/best.pt \
    --out-json results/deeplabv3plus_r50.json
```

## Visualize

```bash
python scripts/visualize.py --config configs/train.yaml \
    --checkpoint runs/deeplabv3plus_r50/best.pt \
    --num-samples 8 --out-dir runs/deeplabv3plus_r50/vis
```

## Results

Filled in after training. Reference numbers are from the original papers,
fine-only, single-scale eval on the Cityscapes val set.

| Model        | Backbone  | Params | val mIoU (ours) | val mIoU (paper) |
|--------------|-----------|-------:|----------------:|-----------------:|
| DeepLabV3+   | ResNet-50 |  ~40M  |             TBD |       ~78 to 80% |
| SegFormer-B0 | MiT-B0    |  ~3.7M |             TBD |       ~76 to 78% |

### Per-class IoU (DeepLabV3+, val)

| road | sidewalk | building | wall | fence | pole | t.light | t.sign | veg | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle |
|------|----------|----------|------|-------|------|---------|--------|-----|---------|-----|--------|-------|-----|-------|-----|-------|------------|---------|
| TBD  |   TBD    |   TBD    | TBD  |  TBD  | TBD  |   TBD   |  TBD   | TBD |   TBD   | TBD |  TBD   |  TBD  | TBD |  TBD  | TBD |  TBD  |    TBD     |   TBD   |

See `docs/experiment_plan.md` for the full setup and the error-analysis plan.

## References

- Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
- Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* (DeepLabV3+), ECCV 2018.
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
- Official Cityscapes scripts: https://github.com/mcordts/cityscapesScripts
