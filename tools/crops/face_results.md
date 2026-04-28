# Face benchmark — MobileFaceNet vs EdgeFace-XS

Method: BlazeFace short-range (same model bundled on-device) detects faces in
each track-tagged person crop, applies 5-keypoint similarity transform to
112×112, both face models embed the aligned face. Cosine matrix → same vs
different identity distributions.

## Headline numbers

| Video | Model | n_faces | same p10 | diff p90 | diff p99 | gap |
|---|---|---:|---:|---:|---:|---:|
| kid_to_wife | MobileFaceNet | 134/221 | +0.147 | +0.548 | +0.678 | **−0.401** |
| kid_to_wife | EdgeFace-XS | 134/221 | +0.139 | +0.527 | +0.809 | **−0.388** |
| boy_indoor_wife_swap | MobileFaceNet | 84/339 | +0.100 | +0.545 | +0.621 | **−0.445** |
| boy_indoor_wife_swap | EdgeFace-XS | 84/339 | −0.180 | +0.466 | +0.535 | **−0.645** |

## Findings

**1. Face is unreliable as a primary signal in this footage.** On
boy_indoor_wife_swap only 84 of 339 person detections produced a face
(~25%). Profile views, occlusion, distant subjects all suppress face
detection. Whatever face model we ship is a *secondary* signal at best.

**2. Face gap is negative across both models, both videos.** Same-person
p10 sits BELOW different-person p90 — no fixed threshold separates same
vs different on these small/varied faces. The on-device wife-vs-boy
face cosine of 0.476 (#102 forensic data) sits in the lower-mid of the
distribution, exactly in the bad band.

**3. EdgeFace-XS does NOT obviously beat MobileFaceNet** with our
current alignment pipeline. Boy_indoor_wife_swap shows a wider
*negative* gap on EdgeFace (−0.645 vs −0.445). EdgeFace was trained
with the standard InsightFace 5-keypoint alignment; our BlazeFace
approximation synthesizes mouth corners from the mouth_center keypoint,
which may be enough to degrade EdgeFace's performance more than
MobileFaceNet's. The paper claims +1.6pp on LFW which we'd need to
reproduce with a proper alignment pipeline to verify.

**4. Boy distribution is intrinsically wide.** boy_indoor_wife_swap
has boy same-sim p10 = +0.100 on MobileFaceNet — meaning 10% of
same-boy face pairs cosine BELOW 0.1. The boy is constantly turning,
hair occluding, expressions changing. Face cannot disambiguate this on
its own.

## Implications

- **Don't swap MobileFaceNet for EdgeFace yet.** The numbers don't
  justify the conversion + alignment work. If we reproduce the LFW
  gain on a known-clean dataset and verify alignment is the bottleneck
  on our footage, revisit.

- **The structural fix (SessionRoster #108) doesn't depend on a better
  face model.** Whichever face model we ship, it adds a *boost* when
  available, not a sole basis for identity. The roster + body
  re-id + score-by-rank do the heavy lifting.

- **Investing in better alignment may help more than swapping models.**
  Switch BlazeFace → MediaPipe FaceMesh (468 landmarks, real mouth
  corners) before swapping the embedder.

- **Face data is sparse on this footage.** A separate axis of work is
  worth scoping: when can face be made more available? Wider
  detection (with both BlazeFace short and full ranges)? Profile-face
  embedders? Skip for now; focus on body.

## Methodology caveat

EdgeFace's published LFW 99.73 vs MobileFaceNet 99.55 implies a small
quality gap on *aligned, frontal, well-lit* faces. Our footage is
none of that. The paper-quality measurement we'd need is on InsightFace's
test set with standardized alignment — a separate spike if face becomes
high-priority. For now, the on-our-footage numbers are what matters
for #102 / #108, and they say "no swap yet."
