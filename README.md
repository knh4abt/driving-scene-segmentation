# Driving Scene Segmentation: How Small Is Too Small for ADAS?

In an autonomous-driving stack, perception models eventually have to run on
a tiny ECU in the car — not on a workstation. The practical question for
that constraint is not "what is the highest mIoU possible." It is **"how
small can the model get before its failure modes become unsafe to ship?"**

This project measures that trade-off concretely for driving-scene
segmentation: two architectures, identical training constraints, an
honest comparison of where each one breaks.

## TL;DR

| Model           | Params | val mIoU   | pixel acc |
|-----------------|-------:|-----------:|----------:|
| DeepLabV3+/R50  |  ~40M  | **72.89%** |    94.99% |
| SegFormer-B0    |  ~3.7M | **60.13%** |    92.75% |

Same dataset (Cityscapes, fine), same seed (42), 100 epochs each, on a
single 12 GB consumer GPU. The 10x parameter reduction trades 12.8 mIoU
points overall — but the trade is **uneven** in a way that matters for
deployment.

## The actual finding

SegFormer's deficit is not distributed evenly across classes. On easy /
common classes (road, sky, car, vegetation) it is only 1-7 points behind.
On the safety-critical rare ones it collapses — most starkly in
**vehicle-vs-vehicle discrimination**:

| Confusion (GT → predicted) | DeepLab | SegFormer |
|----------------------------|--------:|----------:|
| truck → car                |    <1%  | **26%**   |
| bus → car                  |     2%  | **17%**   |
| motorcycle → car           |    <1%  | **16%**   |
| train → bus                |     0%  | **23%**   |

A model that confuses a truck with a car 26% of the time is unshippable
for ADAS: trucks have different braking distances, lane footprint, and
trajectory from cars.
The 10x-fewer-parameters savings come with that very specific cost in
exactly the classes that matter for safety.

Full per-class table and confusion analysis: [`docs/results.md`](docs/results.md).
Failure-case galleries: [`docs/results/failures_deeplab/`](docs/results/failures_deeplab/),
[`docs/results/failures_segformer/`](docs/results/failures_segformer/).

## Reproduce

```bash
conda create -n dss python=3.10 -y && conda activate dss
pip install -r requirements.txt
# point configs/train.yaml:data.root at your local Cityscapes folder

# Train (each takes 4-7 hours on a single 12 GB GPU)
python scripts/train.py --config configs/train.yaml --seed 42
python scripts/train.py --config configs/train.yaml --model segformer_b0 --seed 42

# Evaluate + analyze
python scripts/evaluate.py         --checkpoint runs/deeplabv3plus_r50/best.pt --out-json docs/results/deeplab_eval.json
python scripts/analyze_confusion.py --checkpoint runs/deeplabv3plus_r50/best.pt --out-png docs/results/confusion_deeplab.png
python scripts/failure_gallery.py   --checkpoint runs/deeplabv3plus_r50/best.pt --num-samples 8 --out-dir docs/results/failures_deeplab
```

Cityscapes requires registration: https://www.cityscapes-dataset.com/.
Download `leftImg8bit_trainvaltest.zip` + `gtFine_trainvaltest.zip`, extract
both into one folder, and point `configs/train.yaml:data.root` at it.

## Limitations

This is a one-engineer reproduction on a 12 GB consumer GPU, not a paper run.
The gap between our numbers and published numbers is the deliberate cost of
those constraints:

- **Single seed (42).** Published recipes average over 3 seeds. I report
  the single-seed number honestly rather than cherry-pick.
- **Reduced crop and batch:** 384×384 / batch 4 vs paper's 1024×1024 /
  batch 8. Costs ~5 mIoU points overall and hits SegFormer harder
  (its attention starves on small crops).
- **100 epochs vs paper's 200.** Both val-mIoU curves plateaued by ~epoch
  90, so the marginal gain from 200 epochs is small but not measured.
- **Val-only evaluation.** No cross-dataset test of generalization
  (e.g. BDD100K, ACDC).

A non-trivial training bug worth reading separately (AMP / BatchNorm /
SegFormer interaction): [`docs/debug-notes.md`](docs/debug-notes.md).

## Takeaways

**Inductive biases show up in the failure modes.** CNNs assume locality;
transformers do not. DeepLab keeps truck-vs-car straight using local cues
(wheel geometry, chassis silhouette) — exactly what convolution is built
to capture. SegFormer should disambiguate them via global attention, but
at 384-px crops it cannot see enough of the vehicle. Same task, same
data — different priors, different failures.

**Class imbalance is the hidden enemy.** Every significant confusion
above points toward the larger class — wall → building, truck → car. The
model finds it easier to expand the dominant class than to learn the
rare one. mIoU exists to punish that laziness; the same pattern shows up
in medical imaging, QC, and fraud detection.

**The confusion matrix told a story the headline mIoU could not.**
SegFormer-B0 at 60% is not "a worse DeepLab." It is a model with a
specific vehicle-discrimination weakness — fixable (larger crops,
class-weighted loss, stepping up to SegFormer-B1), but not by
parameter-count alone.

## Repo layout

```
configs/train.yaml          single source of truth for hyperparameters
docs/
  experiment_plan.md        the original plan kept for the record
  results.md                detailed per-class results + confusion analysis
  debug-notes.md            the AMP / BN / NaN failure story
  results/                  eval JSONs, confusion PNGs, failure galleries
src/
  dataset.py                Cityscapes loader + 34→19 trainId remap
  transforms.py             joint image/mask augmentations
  models.py                 build_model("deeplabv3plus_r50" | "segformer_b0")
  metrics.py                confusion-matrix-based mIoU
  utils.py                  color palette, seeding, atomic checkpoint helpers, poly LR
scripts/
  train.py                  train one model end-to-end
  evaluate.py               per-class IoU + JSON dump
  visualize.py              random-sample triptychs
  analyze_confusion.py      row-normalized 19x19 confusion heatmap
  failure_gallery.py        worst-per-image-mIoU triptychs
```

## References

- Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
- Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* (DeepLabV3+), ECCV 2018.
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
- Official Cityscapes scripts: https://github.com/mcordts/cityscapesScripts
