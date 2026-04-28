# Same-vs-different person cosine — full sweep

Method: see `tools/benchmark_reid_models.py`. Track-tagged crops from
YOLOv8 + ByteTrack on the test videos; identity ground truth from
`crops/identity_map.json`. The torchreid OSNet variants are loaded from
the official Hugging Face mirror at
[kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet) (saves us
from the GDrive rate limit on the original torchreid model zoo).

Each torchreid model is run twice:
- `_lb` (letterbox): aspect-preserving resize with gray padding to
  256×128. Matches the on-device CanonicalCropper (#100).
- `_st` (stretch): direct `Resize((256, 128))` — exactly what torchreid
  used in training. Tells us the model's intrinsic ceiling.

The gap metric: `same_p10 − diff_p90`. Positive = the model has a
working threshold for separating same-vs-different at this device's
image distribution. Negative = signals overlap, no fixed threshold
works (the kid_to_wife / #102 failure mode).

## kid_to_wife (n_same=12190, n_diff=12120)

| Model | mode | same p10 | diff p90 | diff p99 | gap |
|---|---|---:|---:|---:|---:|
| MobileNetV3 Large | letterbox | +0.422 | +0.241 | +0.363 | **+0.181** |
| OSNet x1.0 Market (current TFLite) | letterbox | +0.717 | +0.646 | +0.712 | +0.071 |
| OSNet x1.0 MSMT17 | letterbox | +0.572 | +0.544 | +0.605 | +0.029 |
| OSNet x1.0 MSMT17 | **stretch** | +0.659 | +0.576 | +0.638 | **+0.083** |
| OSNet-AIN x1.0 MSMT17 | letterbox | +0.586 | +0.549 | +0.602 | +0.038 |
| OSNet-AIN x1.0 MSMT17 | stretch | +0.622 | +0.569 | +0.622 | +0.053 |
| OSNet-IBN x1.0 MSMT17 | letterbox | +0.532 | +0.527 | +0.579 | +0.005 |
| OSNet-IBN x1.0 MSMT17 | stretch | +0.586 | +0.535 | +0.580 | +0.051 |

## boy_indoor_wife_swap (n_same=32866, n_diff=22080)

| Model | mode | same p10 | diff p90 | gap |
|---|---|---:|---:|---:|
| MobileNetV3 Large | letterbox | +0.174 | +0.372 | -0.198 |
| OSNet x1.0 Market (current TFLite) | letterbox | +0.547 | +0.773 | -0.226 |
| OSNet x1.0 MSMT17 | letterbox | +0.390 | +0.614 | -0.225 |
| OSNet x1.0 MSMT17 | stretch | +0.378 | +0.647 | -0.269 |
| OSNet-AIN x1.0 MSMT17 | letterbox | +0.422 | +0.599 | **-0.177** |
| OSNet-AIN x1.0 MSMT17 | stretch | +0.413 | +0.638 | -0.225 |
| **OSNet-IBN x1.0 MSMT17** | **letterbox** | +0.392 | +0.554 | **-0.162** |
| OSNet-IBN x1.0 MSMT17 | stretch | +0.380 | +0.587 | -0.207 |

## Findings

**No model swap solves the structural overlap.** Both videos still show
overlapping same-vs-different distributions on every variant. The
literature-claimed advantages of cross-domain MSMT17 training,
instance-norm (IBN), and adaptive instance norm (AIN) yield ±5pp gap
shifts on our footage — meaningful but not transformative.

**OSNet-IBN x1.0 MSMT17 is the most consistent improvement** on the
harder scenario (boy_indoor_wife_swap), narrowing the negative gap from
-0.226 (current) to -0.162 (+0.064 improvement). On kid_to_wife
specifically the current Market-trained TFLite still wins (+0.071 vs
+0.005 letterbox), so an unconditional swap would regress the easier
scenario.

**Letterbox vs stretch is a real choice.** For torchreid-trained
models, *stretch* matches their training transform and gives noticeably
better numbers on kid_to_wife (OSNet x1.0 MSMT17: +0.083 stretch vs
+0.029 letterbox). On boy_indoor_wife_swap the relationship inverts —
*letterbox* preserves more identity signal. The on-device
CanonicalCropper currently uses letterbox; if we swap to a torchreid
model, we should consider stretch for that one model and benchmark.

**Engine forensic numbers (#102) cross-validate.** The wife wrong
reacquire scored reId=0.732 against the boy lock. Our benchmark says
boy-vs-wife p90 = 0.646 (current Market), p99 = 0.712. The 0.732
sits above the 99th percentile of cross-person cosines — the outlier
tail that no fixed threshold can reach without rejecting same-person
matches. SessionRoster (#108) addresses this by scoring relative to a
distractor bank instead of an absolute threshold.

## Decision

Don't swap OSNet x1.0 Market until SessionRoster (#108) lands and we
measure residual on real device. If after #108 there's still residual
identity confusion, **OSNet-IBN x1.0 MSMT17 letterbox** is the
recommended swap target — it shows the most robust improvement on the
harder scenarios. Conversion path: PyTorch → ONNX → TFLite via onnx2tf
(same recipe as the current OSNet TFLite). Same model dimensions
(2.7M params, 512-dim output, 256×128 input), so the on-device pipeline
plumbing doesn't need to change.

## Methodology caveats

- **Identity mapping is manual.** Track IDs from ByteTrack don't always
  correspond to identities (ID-switches across reappearances). I labeled
  each track manually from one representative crop. A small number of
  ambiguous tracks ("unknown") were dropped from comparison.
- **boy_indoor_wife_swap has high pose variance.** The boy is constantly
  turning, occluding, changing pose. Same-person variance dominates,
  giving negative gap on every model. This may overstate the structural
  overlap — but it also matches what the engine sees on real footage.
- **CION zoo not evaluated.** README requires signed license + email.
  Per the literature, CION mobile variants would likely give larger
  gains than OSNet variants — worth pursuing if the OSNet-IBN swap
  proves insufficient.
