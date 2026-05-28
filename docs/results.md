# Detailed results

Full per-class numbers and confusion analysis on the Cityscapes val set.
Source artifacts: [`results/deeplab_eval.json`](results/deeplab_eval.json),
[`results/segformer_eval.json`](results/segformer_eval.json),
[`results/confusion_deeplab.png`](results/confusion_deeplab.png),
[`results/confusion_segformer.png`](results/confusion_segformer.png).

## Per-class IoU comparison (%)

| class         | DeepLab | SegFormer |    Δ    |
|---------------|--------:|----------:|--------:|
| road          |   96.86 |     95.35 |   -1.5  |
| sidewalk      |   79.87 |     71.18 |   -8.7  |
| building      |   90.54 |     86.32 |   -4.2  |
| wall          |   45.34 |     35.38 |  -10.0  |
| fence         |   55.85 |     37.31 |  -18.5  |
| pole          |   58.61 |     47.16 |  -11.4  |
| traffic_light |   63.23 |     51.28 |  -12.0  |
| traffic_sign  |   74.45 |     63.00 |  -11.5  |
| vegetation    |   91.74 |     90.22 |   -1.5  |
| terrain       |   60.02 |     55.77 |   -4.3  |
| sky           |   93.01 |     86.10 |   -6.9  |
| person        |   78.61 |     69.67 |   -8.9  |
| rider         |   56.87 |     39.04 |  -17.8  |
| car           |   94.20 |     88.57 |   -5.6  |
| truck         |   70.21 |     31.17 | **-39.0** |
| bus           |   74.55 |     43.42 | **-31.1** |
| train         |   65.40 |     41.58 |  -23.8  |
| motorcycle    |   61.95 |     41.87 |  -20.1  |
| bicycle       |   73.64 |     68.18 |   -5.5  |
| **mIoU**      |**72.89**|  **60.13**|**-12.8** |

Pixel accuracy: DeepLab 94.99% / SegFormer 92.75%.

## Confusion patterns

Both models share the same confusion *shape* — only the magnitude differs.
Top off-diagonal cells from the row-normalized 19x19 matrices:

| Confusion (GT → predicted) | DeepLab | SegFormer |
|----------------------------|--------:|----------:|
| truck → car                |    <1%  | **26%**   |
| bus → car                  |     2%  | **17%**   |
| motorcycle → car           |    <1%  | **16%**   |
| train → bus                |     0%  | **23%**   |
| rider → person             |    12%  |     20%   |
| wall → building            |    23%  |     32%   |
| fence → building           |    15%  |     34%   |
| pole → building            |    14%  |     25%   |
| terrain → vegetation       |    14%  |     23%   |

Two observations:

1. **"Building" is the universal sink.** Every vertical, static object the
   model can't pin down — wall, fence, pole, traffic light — gets dumped
   into "building." Building is the second-largest class with the most
   visual variability, so the model learns it as a catch-all for
   "anything vertical." Both models do this; SegFormer does it ~2x harder.

2. **"Car" is a second sink, but only for SegFormer.** DeepLab keeps
   truck/bus/motorcycle clearly separate from car (diagonal entries
   88% / 79% / 76%). SegFormer collapses them (45% / 55% / 57%). This is
   the single most ADAS-relevant finding in the project — and the
   reason headline mIoU alone hides the safety story.

## Failure-case galleries

Triptychs (image | ground truth | prediction) for the 8 lowest-per-image-mIoU
val samples per model. Filenames embed rank and per-image mIoU
(`rank01_val0316_miou24.97.png` = worst case, val image 316, mIoU 24.97%).

- [`results/failures_deeplab/`](results/failures_deeplab/) — worst-case mIoU floor 25.0%
- [`results/failures_segformer/`](results/failures_segformer/) — worst-case mIoU floor 19.2%

Two images (val[312], val[301]) appear in both models' worst-8 lists —
inherent scene difficulty rather than architecture-specific failure. The
other six per model differ, and each set tells a different story about
where that architecture breaks.
