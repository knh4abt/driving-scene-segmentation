# Training bug worth flagging: SegFormer + AMP + BatchNorm

A non-trivial training failure that took several false starts to diagnose.
Documenting it here because the failure is silent — it produces a
deterministic, plausible-looking mIoU — and the cause is counter-intuitive.

## Symptom

The first SegFormer-B0 training run reached **1.98% val mIoU at epoch 1
and stayed there for 47 epochs**. No errors, no NaN warnings, no exceptions.
tqdm progressed normally, training loss looked vaguely sensible.

1.98% mIoU on 19 classes with `ignore_index=255` is exactly what you get
if a model predicts class 0 ("road") for every pixel of every val image.
So the model had collapsed to predicting one class — the classic
"predict the dominant class" failure mode.

## What I tried first (and why it didn't work)

- **Restored the SegFormer LR** from `3e-5` to the published `6e-5`. The
  config had wrongly halved it on the SGD-style linear-scaling rule;
  AdamW is largely batch-size invariant.
- **Implemented linear warmup.** The config had `warmup_iters: 1500` but
  nothing in `train.py` read it. The freshly-initialised decode head
  needs a gentle ramp.

Both fixes were correct in isolation. The model still hit exactly 1.98%.
Identical to the broken run. That was the first hint the real issue
wasn't "the model didn't learn" — it was something more fundamental.

## Real cause

Inspecting the model output in detail:

```
>>> logits = model(image)
>>> logits.mean(), logits.std(), logits.min(), logits.max()
(nan, nan, nan, nan)
```

All logits NaN. `argmax(NaN)` returns 0, which is "road" in the trainId
mapping. The 1.98% wasn't a model that learned to predict road — it was
an artifact of NaN reduction. The model wasn't outputting anything
coherent at all.

Where the NaN came from:

```
>>> sd['net.decode_head.batch_norm.running_mean']  # NaN
>>> sd['net.decode_head.batch_norm.running_var']   # NaN
```

The BatchNorm running stats in the decode head are corrupt. In eval
mode, BN normalises with `(x - running_mean) / sqrt(running_var)`.
Divide by NaN, get NaN.

The cleanest diagnostic that confirmed it:

```
>>> model.train()  # forces BN to use live batch stats — bypasses running stats
>>> model(image).std()
tensor(0.221)        # healthy
>>> model.eval()
>>> model(image).std()
tensor(nan)
```

Training-mode inference (which doesn't touch the corrupted running
stats) is fine. The model is *not* broken. The running stats are.

## Why the running stats got corrupted

AMP (`torch.cuda.amp`) runs many forward / backward ops in fp16.
Occasionally fp16 overflows on a particular batch — and `GradScaler` is
built exactly for this: it detects NaN/Inf in *gradients* and skips the
optimizer step, so weights stay healthy.

But BatchNorm's running stats are not updated through gradients. They
update inside the forward pass:

```
running_mean = momentum * batch_mean + (1 - momentum) * running_mean
running_var  = momentum * batch_var  + (1 - momentum) * running_var
```

If `batch_mean` or `batch_var` is NaN in any single forward pass
(because some intermediate activation overflowed in fp16), the running
stats permanently become NaN. There is no GradScaler-equivalent for
this update. Training looks fine (loss is computed in train mode, with
live batch stats), but eval-mode inference produces NaN logits forever.

This failure pattern is specific to fragile BN — particularly the
custom decode head BN in SegFormer-B0 at small batch and small crop.
DeepLab's torchvision ResNet BN has been heavily tested with AMP and
is robust.

## Fix

Disable AMP for SegFormer only:

```python
# scripts/train.py
if "segformer" in model_name_lc or "mit" in model_name_lc:
    use_amp = False
```

SegFormer-B0 has ~3.7M parameters, so fp32 fits comfortably in 12 GB at
batch 4. DeepLab keeps AMP — its BN is robust and the larger ResNet
benefits more from the memory savings.

After this fix, Run B trained cleanly to 60.13% mIoU at epoch 98.

## Takeaway

A silent training failure with a deterministic plausible-looking metric
is the worst kind of bug — there is nothing to grep the logs for. The
diagnostic step that finally pinned it: don't trust the metric, inspect
the actual model output. `(logits.mean(), logits.std())` showed NaN in
two seconds; `model.train()` vs `model.eval()` comparison pinned the
issue to running statistics within thirty seconds. Should have done this
on epoch 1 of the first failed run rather than after epoch 47.
