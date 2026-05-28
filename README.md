# Driving Scene Segmentation: DeepLabV3+ vs SegFormer-B0 on Cityscapes

**Question:** For driving-scene segmentation, can a modern transformer
(SegFormer-B0, ~3.7M params) match a classical CNN (DeepLabV3+/R50, ~40M
params) under the kind of compute constraints a real embedded deployment
would impose?

## TL;DR

| Model           | Params | val mIoU   | pixel acc | epochs | crop | batch |
|-----------------|-------:|-----------:|----------:|-------:|-----:|------:|
| DeepLabV3+/R50  |  ~40M  | **72.89%** |    94.99% |    100 |  384 |     4 |
| SegFormer-B0    |  ~3.7M | **60.13%** |    92.75% |    100 |  384 |     4 |

Both trained on a single 12 GB consumer GPU with identical data, augmentation,
and seed (42). Reference paper numbers (1024x1024 crops, batch 8, full
recipe): DeepLabV3+ ~78-80%, SegFormer-B0 ~76-78%. The gap between paper
numbers and ours is the deliberate cost of working in 12 GB; see
[Limitations](#limitations).

## The actual finding

SegFormer-B0 trades 12.8 mIoU points overall, but the trade is **uneven**:

- Easy / common classes (road, sky, vegetation, car, building): only 1-7 pts behind.
- Rare / hard classes (truck, bus, train, fence, rider, motorcycle): **17-39 pts behind**.

The confusion matrices show *why*. Both models make the same kinds of mistakes
(the failure shape is shared); SegFormer makes them ~2x more severely.
*Plus* SegFormer has one extra failure mode DeepLab does not: it confuses
trucks, buses, and motorcycles with cars.

| Confusion (GT class → predicted class) | DeepLab | SegFormer |
|----------------------------------------|--------:|----------:|
| truck → car                            |    <1%  | **26%**   |
| bus → car                              |     2%  | **17%**   |
| motorcycle → car                       |    <1%  | **16%**   |
| train → bus                            |     0%  | **23%**   |
| rider → person                         |    12%  |     20%   |
| wall → building                        |    23%  |     32%   |
| fence → building                       |    15%  |     34%   |

For an ADAS framing this matters: truck/bus/motorcycle/rider drive braking,
occupancy, and trajectory decisions. The 10x-fewer-parameters savings of
SegFormer-B0 come with a very specific cost in exactly the safety-critical
classes.

Full 19x19 heatmaps:
- [`docs/results/confusion_deeplab.png`](docs/results/confusion_deeplab.png)
- [`docs/results/confusion_segformer.png`](docs/results/confusion_segformer.png)

## Task

For every pixel of a 1024x2048 street image, predict one of 19 classes:
road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle,
bicycle.

The metric is mean Intersection over Union:

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \qquad \text{mIoU} = \frac{1}{19}\sum_{c=1}^{19} \text{IoU}_c$$

Pixel accuracy is not the headline metric: road alone covers ~35% of pixels
in a typical scene, so a model predicting only road gets ~30% accuracy for
free. mIoU averages per-class IoU equally and punishes that degenerate
solution.

## Per-class results

Per-class IoU (%) on the Cityscapes val set, both models.

| class         | DeepLab | SegFormer |    Δ   |
|---------------|--------:|----------:|-------:|
| road          |   96.86 |     95.35 |  -1.5  |
| sidewalk      |   79.87 |     71.18 |  -8.7  |
| building      |   90.54 |     86.32 |  -4.2  |
| wall          |   45.34 |     35.38 | -10.0  |
| fence         |   55.85 |     37.31 | -18.5  |
| pole          |   58.61 |     47.16 | -11.4  |
| traffic_light |   63.23 |     51.28 | -12.0  |
| traffic_sign  |   74.45 |     63.00 | -11.5  |
| vegetation    |   91.74 |     90.22 |  -1.5  |
| terrain       |   60.02 |     55.77 |  -4.3  |
| sky           |   93.01 |     86.10 |  -6.9  |
| person        |   78.61 |     69.67 |  -8.9  |
| rider         |   56.87 |     39.04 | -17.8  |
| car           |   94.20 |     88.57 |  -5.6  |
| truck         |   70.21 |     31.17 |**-39.0**|
| bus           |   74.55 |     43.42 |**-31.1**|
| train         |   65.40 |     41.58 | -23.8  |
| motorcycle    |   61.95 |     41.87 | -20.1  |
| bicycle       |   73.64 |     68.18 |  -5.5  |
| **mIoU**      |**72.89**|  **60.13**|**-12.8**|

Worst per-image mIoU on the val set:
- DeepLab: 25.0%
- SegFormer: 19.2%

Two images (val[312], val[301]) appear in *both* models' worst-8 lists,
which suggests inherent scene difficulty rather than architecture-specific
failure. The other six per model differ — each architecture has its own
weakness beyond the shared hard cases. Triptychs (image | ground truth |
prediction) for the worst 8 of each model live in:

- [`docs/results/failures_deeplab/`](docs/results/failures_deeplab/)
- [`docs/results/failures_segformer/`](docs/results/failures_segformer/)

## Repo layout

```
configs/
  train.yaml           single source of truth for hyperparameters
docs/
  experiment_plan.md   the original plan (kept for the record)
  results/             eval JSONs, confusion matrices, failure galleries
src/
  dataset.py           Cityscapes loader + 34->19 trainId remap
  transforms.py        joint image/mask augmentations
  models.py            build_model("deeplabv3plus_r50" | "segformer_b0")
  metrics.py           confusion-matrix-based mIoU
  utils.py             color palette, seeding, atomic checkpoint helpers, poly LR
scripts/
  train.py             train one model end-to-end
  evaluate.py          per-class IoU + JSON dump
  visualize.py         random triptychs (image | GT | prediction)
  analyze_confusion.py row-normalized 19x19 confusion-matrix heatmap
  failure_gallery.py   worst-per-image-mIoU triptychs (ranked)
runs/
  <experiment>/
    best.pt last.pt    + .prev rotations from the atomic save
    vis/               random-sample triptychs
requirements.txt
```

## Dataset setup

Cityscapes requires registration.

1. Create an account at https://www.cityscapes-dataset.com/.
2. Download from the Downloads page:
   - `leftImg8bit_trainvaltest.zip` (~11 GB) — RGB images
   - `gtFine_trainvaltest.zip` (~241 MB) — fine annotations
3. Extract both into one folder, e.g. `D:/datasets/cityscapes`:

   ```
   cityscapes/
     leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
     gtFine/{train,val,test}/<city>/*_gtFine_labelIds.png
   ```

4. Set `data.root` in `configs/train.yaml` to that folder.

Only the fine annotations are used. Published numbers for DeepLabV3+ and
SegFormer are reported on the fine set, so this matches the literature.

## Environment

```bash
conda create -n dss python=3.10 -y
conda activate dss
pip install -r requirements.txt
```

Tested on Windows 11 + CUDA 12.1 + PyTorch 2.4, single 12 GB GPU.

## Train

```bash
# DeepLabV3+ / ResNet-50 (default)
python scripts/train.py --config configs/train.yaml --seed 42

# SegFormer-B0
python scripts/train.py --config configs/train.yaml --model segformer_b0 --seed 42

# Resume an interrupted run from runs/<experiment>/last.pt
python scripts/train.py --config configs/train.yaml --resume
```

Checkpoints are saved atomically (`.tmp` -> `fsync` -> rename) and the previous
checkpoint is kept as `.prev`, so a crash mid-save cannot leave a corrupt-or-
missing checkpoint. Resume falls back to `.prev` if `last.pt` is unreadable.

## Evaluate and analyze

```bash
# per-class IoU table + JSON dump
python scripts/evaluate.py --config configs/train.yaml \
    --checkpoint runs/deeplabv3plus_r50/best.pt \
    --out-json docs/results/deeplab_eval.json

# row-normalized confusion matrix as a PNG heatmap
python scripts/analyze_confusion.py --config configs/train.yaml \
    --checkpoint runs/deeplabv3plus_r50/best.pt \
    --out-png docs/results/confusion_deeplab.png

# 8 worst-per-image-mIoU triptychs (ranked, mIoU embedded in filename)
python scripts/failure_gallery.py --config configs/train.yaml \
    --checkpoint runs/deeplabv3plus_r50/best.pt \
    --num-samples 8 --out-dir docs/results/failures_deeplab

# random-sample triptychs (fixed seed = same indices across models)
python scripts/visualize.py --config configs/train.yaml \
    --checkpoint runs/deeplabv3plus_r50/best.pt \
    --num-samples 8 --out-dir runs/deeplabv3plus_r50/vis
```

## Limitations

This is a one-engineer reproduction on a 12 GB consumer GPU, not a paper run.
The gap between our numbers and the published numbers is the deliberate cost
of those constraints. Each item below is a knob a full run would change:

- **Single seed (42), no averaging.** Published numbers are usually averaged
  over 3 seeds. I report the single-seed number honestly rather than cherry-pick.
- **Reduced crop and batch:** 384x384 / batch 4 vs paper's 1024x1024 / batch 8.
  This roughly halves the per-iter signal and disproportionately hurts
  SegFormer, which depends on global context that small crops starve.
- **100 epochs vs paper's 200.** Both runs' val-mIoU curves had plateaued by
  ~epoch 90, so the marginal gain from 200 epochs is small — but not measured.
- **Val-only evaluation.** No cross-dataset test of generalization. Would need
  BDD100K or ACDC to test across geography / weather.

### A bug worth flagging

Run B initially collapsed to predicting "road" for every pixel — a
deterministic 1.98% mIoU across many epochs. Diagnosis: AMP/fp16 overflow
in SegFormer's decode-head BatchNorm. `GradScaler` protects optimizer steps
from NaN gradients, but BN running stats update during the forward pass
(not via gradients), so a single fp16 overflow permanently corrupted them.
Eval-mode inference (which uses running stats) then divided by NaN and
produced NaN logits everywhere; `argmax(NaN)` returns index 0, which happens
to be "road."

Fix: disable AMP for SegFormer (see `scripts/train.py`). SegFormer-B0 is
small enough that fp32 fits in 12 GB comfortably. DeepLab keeps AMP — its
ResNet BN is robust to it.

Diagnostic that pinned the cause: run inference in train mode (uses live
batch stats → healthy logits) vs eval mode (uses running stats → NaN), and
inspect `state_dict()['decode_head.batch_norm.running_var']` directly.
Worth flagging for anyone training SegFormer on a small GPU with AMP enabled.

## What I took away from this

A few things this project clarified that I'll carry into the next one.

**Inductive biases matter, and they show up in the failure modes.** CNNs
assume locality and translation equivariance; transformers assume nothing.
DeepLab keeps truck-vs-car straight because the local cues (wheel geometry,
chassis silhouette) are exactly what convolution is built to capture.
SegFormer's global attention should help with that too, but at 384-px crops
it doesn't see enough vehicle context. Same task, same data — different
priors, different failures. The question that generalizes: *what does this
architecture assume, and is the assumption appropriate for my data and my
compute?*

**Class imbalance is the hidden enemy.** Every significant off-diagonal
confusion above points the same direction: toward the larger class. Wall,
fence, pole, traffic-light all collapse into "building." Truck, bus,
motorcycle collapse into "car." The model finds it easier to expand the
dominant class than to learn the rare one. mIoU exists to make this kind of
laziness visible. The same pattern shows up wherever class frequencies are
skewed — medical imaging (tumors are tiny), QC (defects are rare), fraud
(legitimate is the overwhelming default).

**Hardware constraints are the work, not a footnote to it.** This project
is a Pareto trade-off study: mIoU vs parameter count. Every applied ML
problem I'll touch in industry will be one — accuracy vs latency, accuracy
vs memory, accuracy vs labeled-data needed. Documenting which constraints
I chose and what they cost is the engineering. "How high did you go" is
mostly fiction outside of papers.

**A confusion matrix tells a story the headline mIoU can't.** SegFormer-B0
at 60% sounds like a generically weaker version of DeepLab at 73%. The
matrix shows the 60% is held together by near-perfect performance on common
classes, with the deficit concentrated in vehicle-vs-vehicle confusion.
That changes the conclusion: SegFormer-B0 isn't a worse DeepLab. It's a
model with a *specific* class-discrimination weakness that's bad for ADAS
— and that's addressable: larger crops, class-weighted loss, focal loss,
or stepping up to SegFormer-B1.

## References

- Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
- Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* (DeepLabV3+), ECCV 2018.
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
- Official Cityscapes scripts: https://github.com/mcordts/cityscapesScripts
