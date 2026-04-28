# Baseline same-vs-different person cosine — current shipping models

Method: see `tools/benchmark_reid_models.py`. Track-tagged crops from
YOLOv8 + ByteTrack on the test videos, identity ground truth from
`crops/identity_map.json`, ImageNet-normalized letterbox input for OSNet.

## Headline numbers

| Video | Model | same p10 | diff p90 | diff p99 | gap (p10 − p90) |
|---|---|---|---|---|---|
| kid_to_wife | MobileNetV3 Large | +0.422 | +0.241 | +0.363 | **+0.181** |
| kid_to_wife | OSNet x1.0 | +0.717 | +0.646 | +0.712 | **+0.071** |
| boy_indoor_wife_swap | MobileNetV3 Large | +0.174 | +0.372 | +0.491 | **−0.198** |
| boy_indoor_wife_swap | OSNet x1.0 | +0.547 | +0.773 | +0.858 | **−0.226** |

## What the numbers say

**The structural problem is real, not a calibration miss.** On
boy_indoor_wife_swap both shipping models produce a NEGATIVE gap — the
worst 10% of same-person pairs sit below the best 10% of different-person
pairs. There is no fixed cosine threshold that separates same from
different on this clip with either model. This is the kid_to_wife
failure mode generalized: relative ranking, not absolute thresholding,
is the only axis on which we can win.

OSNet x1.0 on kid_to_wife specifically: same_p10 (0.717) is barely above
diff_p99 (0.712). At threshold 0.715 you'd accept ~50% of same-person
matches and reject ~99% of different-person. Increase the threshold
and you lose more same-person matches faster than you gain different-
person rejections.

## Per-identity-pair detail (kid_to_wife)

OSNet x1.0:
- boy vs wife: p10=0.526 p50=0.573 p90=0.646
- boy same: p10=0.727 p50=0.821
- wife same: p10=0.677 p50=0.837

The boy-vs-wife distribution's TAIL (p90=0.646) is what causes wrong
reacquires. Engine-logged sim=0.732 (#102 forensic) sits in that tail.

MobileNetV3 Large:
- boy vs wife: p10=0.053 p50=0.156 p90=0.241
- boy same: p10=0.508 p50=0.654
- wife same: p10=0.259 p50=0.577

Wider bands but same overlap pattern — wife's same-sim p10=0.259 sits
BELOW boy-vs-wife p90=0.241 means a fixed threshold can't separate
"is this candidate the wife I locked on" from "is this candidate the
boy I'm trying to track" reliably.

## Implications

1. **A better backbone helps but does not solve.** The literature
   suggests CION zoo and OSNet-AIN add ~5-12% mAP cross-domain — that
   would shift the diff_p90 down by some amount and lift same_p10 by
   some amount. But the *structural* property (overlap exists) is
   likely to remain because both shipping models already show it on a
   real-footage dataset.

2. **SessionRoster ranking (#108) is the right fix regardless of
   model.** Identity-by-rank works even when distributions overlap, as
   long as the candidate's CORRECT slot tends to outscore the WRONG
   slot. That's a much weaker condition than a fixed threshold.

3. **Model upgrade still worthwhile.** A wider gap reduces the rate at
   which the lock slot is the second-best match — i.e. it reduces the
   load on the margin gate.
