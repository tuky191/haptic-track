# HapticTrack Identity Research: Two-People & Two-Vehicles Failures

Author: research pass on master @ commit `a7f5612`
Scope: identity-tier failures observed live (issues #51, #62)
Out of scope: detector quality, drift killers, FTFTracker IDs (#65)

---

## 0. TL;DR

The identity stack is **architecturally sound but practically blind on people** because the only embedder that meaningfully participates in the cascade gates is `mobilenet_v3_large_embedder.tflite` — a generic ImageNet classifier, not a re-ID model. OSNet (which **is** a real re-ID model) and MobileFaceNet are loaded and the per-frame plumbing is in place, but they are only used in **ranking among survivors**, not in **gating**. Every person candidate first has to pass a generic-MobileNetV3 `MIN_EMBEDDING_SIMILARITY = 0.15` floor + adaptive floor up to `0.5`, and on the live person session every same-person candidate scored 0.08–0.34 there.

> **Concrete: in `session_20260424_203704_person`, FrameToFrameTracker was correctly matching the same physical person (`id=234`, IoU 70–88%) across 50+ frames. ReacquisitionEngine rejected every single one of them with the message `(hard threshold)` — meaning it never even got far enough into `scoreCandidate()` to consult OSNet, MobileFaceNet, or the online classifier.** The whole identity stack downstream of the generic embedder never executed.

The same session immediately afterwards on `cup` produced sim 0.70–0.91 and reacquired in 1–11 frames. So the system works fine when MobileNetV3-Large can produce a discriminative similarity, and falls off a cliff when it can't — and it can't on people because that's not what it was trained to do.

The recommended sequence at the bottom is:

1. **PR1**: Use OSNet as the gating embedder for persons (it's already computed, just promoted from ranking-only to gate). Expected to recover a large fraction of two-people failures, ~1 day.
2. **PR2**: Swap the generic embedder from MobileNetV3-Large (ImageNet-classification) to a self-supervised feature extractor (DINOv2-small or a distilled MobileCLIP), which is much stronger on instance-level identity for non-person objects (cars, bottles). 3–5 days because of TFLite conversion.
3. **PR3**: Replace the `(min, adaptive_floor)` constants with a per-lock learned threshold derived from the gallery's intra-class spread + scene negatives at lock time. 1–2 days.

---

## A. What our current system actually does (and where it falls short)

The identity tier loads **five** ML models. I'll describe each by what it actually contributes, then walk through how `scoreCandidate()` consumes them.

### A.1 The models

| Model | Asset | Dim | Purpose **as designed** | Purpose **in our cascade** |
|---|---|---|---|---|
| MobileNetV3-Large embedder | `mobilenet_v3_large_embedder.tflite` | 1280 | ImageNet classification feature extractor, repurposed for generic visual similarity via L2-normalized penultimate activations | **Gating signal for everything** — the only embedding consulted by `MIN_EMBEDDING_SIMILARITY`, the adaptive floor, and the position/size override path |
| OSNet x1.0 (Market-1501) | `osnet_x1_0_market.tflite` | 512 | Person re-ID across cameras (Market-1501 trained: 1,501 IDs, omni-scale residual) | Ranking only — only used among gate survivors, weighted 0.40 in `hasReId` branch |
| MobileFaceNet | `mobilefacenet.tflite` | 192 | Face identity (ArcFace/InsightFace family, MS-Celeb-1M-style training) | Ranking only — weighted 0.45 when face is present, otherwise unused |
| Crossroad-0230 | `person_attributes_crossroad_0230.tflite` | 8 binary + 2 colors | OpenVINO Open Model Zoo person-attribute classifier (gender, hat, bag, ...). Trained on Market-1501-style attribute annotations | Ranking only, weight 0.05–0.10. Provides `PersonAttributes.similarity()` which is `0.6 * (1 - L1 prob diff) + 0.4 * color_match` |
| BlazeFace + age-gender-retail-0013 | `blaze_face_short_range.tflite` + `age_gender_retail_0013.tflite` | gender (softmax), age (regress) | Face detect + gender/age classify | Face crops feed MobileFaceNet; gender overrides Crossroad-0230 inside `PersonAttributes` |

So: **four models contribute identity information, one model gates on it.**

### A.2 Lock-time gallery construction

`AppearanceEmbedder.embedWithAugmentations()` (lines 113–141) generates **5** embeddings:

1. Original masked crop (segmenter output, fallback to raw crop)
2. Rotated 90° CW
3. Rotated 180°
4. Rotated 270° CW (= 90° CCW)
5. Horizontal flip

In addition `ObjectTracker.lockOnObject()` (lines 226–268) computes:

- One color histogram (`computeColorHistogram` over masked crop)
- Person attributes (Crossroad-0230 + face), if `label == "person"`
- One OSNet re-ID embedding, if person
- One MobileFaceNet face embedding, if person AND a face was detected within the crop
- Scene negatives: one MobileNetV3 embedding per *other* detection visible at lock time

During tracking, `processBitmapInternal` accumulates one new MobileNetV3 embedding per ~15 confirmed VT frames, gated by a centroid-similarity diversity check (don't add if `centroidSim > 0.92` and gallery has ≥8 entries). Gallery is capped at `MAX_GALLERY_SIZE = 12`. **Crucially: only the generic MobileNetV3 embedding is updated/gallery-stored — OSNet and the face embedding are stored once and never refreshed**. Face embedding is added progressively once *if* the lock-time face detection failed but a face later becomes visible (`reacquisition.addFaceEmbedding`, ObjectTracker line 435).

> **Bug-flag, not a critical one**: the gallery is updated based on `centroidSim`, which is the **gallery's own** centroid. There's no comparison to the lock embedding specifically. When VT drifts slowly and template self-verification hasn't tripped yet, the centroid drifts with the gallery — so the diversity check can keep accepting drifted embeddings as "diverse." Probably contributes to the multi-second sim decay we sometimes see in chair sessions but is not the dominant problem.

### A.3 Are OSNet and MobileFaceNet used in gating or only in scoring?

**Only in scoring.** Look at `ReacquisitionEngine.scoreCandidate()` (lines 494–695):

```
GATE 1  candidate.embedding == null while gallery exists      → reject
GATE 2  appearanceScore < adaptiveFloor (0.3..0.5)            → reject + neg example
        ↓ (geometric override = appearanceScore > 0.55)
GATE 3  position distance > posThreshold && !geometricOverride → reject
GATE 4  size ratio > sizeThreshold && !geometricOverride       → reject
        ↓ (label override = appearanceScore > 0.7)
GATE 5  candidateIsPerson != lockedIsPerson && !labelOverride → reject + neg example

RANK    if (hasFace)        face*0.45 + reId*0.20 + emb*0.10 + ...
        elif (hasReId)      reId*0.40 + emb*0.20  + ...
        elif (hasAppearance) emb*0.30/0.50 + cls*0.25 + margin*0.20 + ...
        else                 position/size only
```

`appearanceScore` is **always** `bestGallerySimilarity(candidate.embedding)` against the MobileNetV3 gallery. If that's below floor, the function returns `null` *before* OSNet, MobileFaceNet, attributes, color, or position even matter.

This is the structural bug behind the live data. Look at the rejected lines from the person session:

```
[Reacq]   rejected id=234 label="person" sim=0.243 (hard threshold)
[Reacq]   rejected id=234 label="person" sim=0.146 (hard threshold)
[Reacq]   rejected id=234 label="person" sim=0.270 (hard threshold)
... 97 such lines, all sim ∈ [0.08, 0.34], same physical id ...
```

Every single rejection hit `MIN_EMBEDDING_SIMILARITY` (line 525) or the adaptive floor (line 524). OSNet `reIdEmbedding` was being computed by the async pipeline (`computeEmbeddingsSync` runs on top-2 person candidates) — and then thrown away unread because the candidate didn't pass GATE 2. The OSNet similarity for that same person against the lock-time OSNet embedding might well have been 0.7+, but the engine never asked.

### A.4 Color histogram + person attributes interaction

Both are scored — **never gated**. In the `hasReId` ranking branch:

- color: weight 0.15
- attrs: weight 0.10

Color histogram uses HSV with V deliberately excluded (`EmbeddingUtils` line 131–164). This is correct for lighting invariance but means a black T-shirt and a black jacket are nearly identical vectors. Person-attributes `similarity()` mixes raw probability L1 distance (0.6 weight) and color name match (0.4). The clothing color is the cleanest signal for person discrimination but is collapsed into `0.4 * (boolean upper match) + 0.4 * (boolean lower match)` — a step function — and then weighted 0.10 in the final score. Two people in a blue shirt vs a red shirt get only `0.10 * 0.4 = 0.04` of separation from this signal, which is hopeless against the embedder noise.

### A.5 Failure mode mathematically (two people)

Set up: locked subject embedding gallery = `g₁ … g₉` (MobileNetV3, 1280-dim). Two visible person candidates B₁ (the locked person) and B₂ (a different person). What `scoreCandidate` does:

```
sim_B1 = max_i cosine(g_i, embed(B1))     # we observed 0.08–0.34
sim_B2 = max_i cosine(g_i, embed(B2))     # likely similar range

floor = clamp(minGallerySim * 0.75, 0.3, 0.5)
        with galleryMature → max(floor, 0.4)
```

Because MobileNetV3-Large is trained for *category* discrimination, not identity, two persons in different clothes against a busy indoor backdrop produce embeddings whose cosine to ANY of the lock gallery's augmented views can sit at 0.1–0.35. Both B₁ and B₂ fail the `0.4` floor. **Both rejected, both added as scene negatives, classifier never gets a positive new sample.**

Worse, since both are being added as negatives, the prototype margin and online classifier *learn that all persons are negatives* — exactly the failure mode the discriminative scoring was supposed to prevent.

For the cars scenario it's a milder version of the same problem: same category, similar texture, MobileNetV3-Large was trained to map "car" to "car" not to distinguish "red sedan A" from "red sedan B." Color histogram has an effect (because colors actually differ) but only as a soft scoring signal on candidates that have already passed the embedding floor.

### A.6 Other small issues found while reading

- **The `cup` session reacquired on `id=174 label="cake"` with `sim=0.907`** (line 58 of cup log). That's the label-override path firing as designed, but it's also a mild example of MobileNetV3 similarity > 0.7 across categories — suggesting the override threshold isn't quite right either.
- The **adaptive floor** (line 523) uses `_minGallerySim * 0.75` clamped `[0.3, 0.5]`. But `_minGallerySim` is computed only over the *augmented + accumulated* gallery, which becomes very tight after VT runs for a while (all from very similar viewpoints). For person crops the augmented entries are themselves not very similar to each other — rotated/flipped person crops embed quite differently — so on lock the floor gets clamped at `0.3` and stays there. This is the only thing keeping persons from being rejected entirely; without it, even the adaptive system can't help.
- **GATE 1 (line 513) is correct but punishing for fast scenarios**: a person leaving the frame for half a second produces no async embedding for the candidate when they re-enter, and the sync fallback only fires for the *single closest* same-category candidate (`needsSync.minByOrNull` line 672). With two persons in scene this can systematically embed the wrong one.
- **Scene-negative poisoning** (line 526–527 and 582): rejected candidates are added as negatives. When the locked person is scored low (sim 0.15) they get added as a negative and the centroid+classifier learn the locked person is "not us" — the sim-0.85 filter on `addSceneNegative` (line 147) doesn't help, that path is only used during *confirmed* tracking, not during search. **This is a serious self-poisoning bug for the failure case.** The system is actively training itself to reject the right answer.

---

## B. State of the art — on-device visual identity

### B.1 Generic appearance embedders

The job MobileNetV3-Large is doing in our pipeline (generic visual similarity) is now done much better by self-supervised foundation models.

**DINOv2** (Meta, 2023; updated through 2024). Self-supervised ViT trained on LVD-142M images. Without any fine-tuning, the [CLS] token features perform very strongly on instance retrieval and re-identification. Distilled variants:
- `dinov2-small` (ViT-S/14): 21M params, 384-dim feature, ~85MB FP32 / ~22MB INT8.
- `dinov2-base` (ViT-B/14): 86M params, 768-dim feature, too big for live inference.
- A 14×14 patch model at 224×224 is ~256 patch tokens, manageable on Adreno 740 in FP16 if you drop to 196×196 or 168×168 input.
- **TFLite availability**: there are community PyTorch→ONNX→TFLite conversions (huggingface community models). Not a one-line drop-in. Expect 1-2 days of conversion + delegate work. There's no official TFLite release.

**SigLIP / SigLIP-2** (Google, 2024). Sigmoid-loss CLIP variant, same architecture family, generally cleaner image features than CLIP at the same parameter count. SigLIP-Base is still 200MB+ — `siglip-base-patch16-224` is too heavy. SigLIP-So400m is enormous. No suitable on-device variant ships from Google.

**MobileCLIP** (Apple, CVPR 2024). Designed explicitly for on-device. Three sizes: S0 (3.1M params), S1 (21.5M), S2 (35.7M). MobileCLIP-S2 has been shown competitive with CLIP-ViT-B/16 on retrieval. **Apple released CoreML, not TFLite, but the architecture (MCi backbones) is convertible**. There's a community ONNX-to-TFLite path. ~50–100MB FP16 for S2.
- This is the most plausible drop-in replacement for MobileNetV3 in our pipeline. Output is L2-normalizable, ~512-dim, generic visual similarity, demonstrably better at instance-level discrimination than ImageNet classifiers.

**EVA-02 / OpenCLIP-tiny / TinyViT**. EVA-02 is too heavy. OpenCLIP has a `coca_ViT-B-32`, still big. TinyViT-21M is small but the embedding quality on identity tasks is mediocre — basically a smaller distilled CLIP, similar regime to MobileCLIP-S0/S1 but worse-trained. Probably not worth pursuing over MobileCLIP.

**Recap for our stack**: the most practical generic embedder upgrades, ranked by ease-of-graft:
1. **MobileCLIP-S2** (PyTorch → ONNX → onnx2tf → TFLite, MediaPipe ImageEmbedder will not load it; need the same raw-TFLite GPU pattern we already use for `magic_touch.tflite`).
2. **DINOv2-small distilled** (similar conversion pain, larger model, better features for fine-grained instance retrieval).
3. **MobileNetV3-Large staying as fallback**.

### B.2 Person re-ID

The state of the art moved significantly between 2019 (when OSNet shipped) and 2024.

**TransReID (ICCV 2021)** and follow-ups. Pure-ViT person re-ID. Significantly better cross-camera mAP than OSNet on Market-1501 / DukeMTMC, but the base model is 86M params (ViT-B/16) — too big for live use. There's TransReID-SSL (self-supervised pretraining) which improves further but doesn't change the size. Not on-device-friendly.

**CLIP-ReID (AAAI 2023)**. Uses CLIP image features + learned text prompts as a re-ID anchor. Numbers strong, model still 86M+. Off-device unless distilled.

**AGW / BoT baselines (Luo et al., 2019)**. Strong CNN-based baselines, often used as the "production-realistic" comparison point. ResNet-50 backbone + BNNeck + Triplet+ID loss. Comparable to OSNet x1.0, no clear advantage.

**OSNet variants**: x0.25, x0.5, x0.75, x1.0, x1.0-IBN, x2.0. We're on x1.0 (4.2MB, 256×128 input, 512-dim). The IBN variant (Instance + Batch Norm) is 30% better on cross-domain mAP for similar size — would be a free upgrade if a TFLite conversion exists. The Torchreid model zoo has `osnet_ibn_x1_0` as a PyTorch checkpoint; same conversion path as the existing OSNet would produce a comparable TFLite.

**LightMBN, Cluster-Contrast, CLIP-ReID-distilled**. All exist; all need PyTorch→TFLite conversion. None has a public TFLite release that I can confirm.

**Practical answer**: OSNet x1.0 is roughly the right model for the budget; we're underusing it because we never gate on it. The bigger win is consulting it in the gate, not replacing it. A swap to OSNet-IBN-x1.0 for a 5–15% mAP bump is plausible and cheap once the conversion script exists.

### B.3 Vehicle re-ID

This is a thinner field on-device.

**VeRi-776, VRIC, VehicleID**: standard datasets. Models that train on these and produce strong embeddings include VEReID (vehicle re-ID transformer), MGN (Multiple Granularity Network) variants, and PCB-style strip-based re-ID.

**OpenVINO has a `vehicle-reid-0001`** (~4MB ONNX, ~256-dim) — designed for the exact "is this the same red car?" case. Same provenance as our `person_attributes_crossroad_0230` (Intel OMZ). Could be added with the same TFLite conversion + raw-TFLite GPU pattern. Trained on VeRi-776, VRIC, and proprietary city data.

**Real-world generalization**: vehicle re-ID models trained on surveillance data tend to overfit to dataset-specific orientations (back-of-vehicle license plate views, side profiles). For handheld phone footage (mixed angles, motion blur, kid filming dad's car), generalization is uncertain. There's no TFLite-friendly model I'm confident generalizes well to in-the-wild filming.

**Alternative**: don't ship a vehicle-specific model. Instead, rely on a stronger **generic** embedder (DINOv2 / MobileCLIP) which has been shown to handle vehicle instance retrieval reasonably well as a side effect of broad pretraining. This is what video tracking systems (SAM2, Cutie) do — they don't have category-specific re-ID at all, they just have very strong pretrained features.

### B.4 General object re-ID / video object tracking

**SAM2 (Meta, 2024)**. Segment Anything 2 with memory bank. Per-frame segmentation conditioned on a memory bank of prior masks. Identity is implicit: the memory bank provides cross-frame consistency. SAM2 small variant is 39M params, still too heavy for 30fps live. There is no on-device SAM2 release.

**Cutie (CVPR 2024)**. Object-level memory reading for video segmentation. Encoder is ResNet-50 + transformer. Comparable size to SAM2-tiny. Not on-device.

**SAMTrack, XMem++, DeAOT**. All in the "memory bank video segmentation" family. Same scale issues.

**Take-away**: the SoTA video object trackers don't use a separate identity classifier — they use a *memory bank* of features from past frames + a transformer that attends over the memory bank. Our `_embeddingGallery` is a primitive version of the same concept. The way to leverage this idea on-device is:

- Keep more diverse exemplars in the gallery (multi-view, multi-pose).
- Use attention-style aggregation instead of `max(cosine)` — e.g. soft-max-weighted nearest-neighbor.
- At gate time, use the *combined* gallery distribution (mean + variance) not the single best match.

These are 1-2 day algorithmic improvements that don't need a new model. They directly target the failure mode where one bad augmented embedding dominates `bestGallerySimilarity`.

### B.5 Hybrid approaches / fusion

The face + body fusion we already have is the textbook approach (e.g. ABDNet, hybrid PFC+face). What's missing:

- **Reciprocal nearest neighbors / k-reciprocal re-ranking** (Zhong et al., CVPR 2017): a re-ID re-ranking trick that significantly improves retrieval mAP at zero extra inference cost. Re-ranks the top-k candidates by checking whether the locked object is also a near neighbor of each candidate. Practical on-device because the gallery is tiny.
- **Part-based attention**: don't pool the whole person crop into one embedding. Instead embed body parts separately. PCB / MGN do this. Significantly more robust to occlusion (someone walking behind a chair). Heavyweight to implement; would need a multi-output OSNet variant or custom striping post-process.
- **Metric learning on top of pretrained features**: learn a small MLP head at lock time on (gallery vs scene negatives) embeddings. Our `OnlineClassifier` already does this — but only for scoring, not for gating, and it requires negatives to train, which the failure case can't generate (no positives ever pass the floor).

**Exemplar SVMs** (Malisiewicz, 2011) — train a tiny linear SVM at lock time, exactly one positive (the lock crop's gallery), against a fixed pool of generic negatives bundled with the app. This is the right structural fix for "we can't generate negatives at search time because the floor is too high." Ship a few thousand frozen negative embeddings as an asset, train the SVM at lock, gate on SVM score instead of cosine floor. ~50KB of pre-computed negative MobileCLIP embeddings could ship in assets.

---

## C. Concrete improvement levers, ranked

I'll be explicit about which of these contradict each other and which compose. Most compose.

### C.1 Use OSNet as the primary identity gate for persons (highest leverage)

**Summary**: When `lockedIsPerson` is true and a candidate has `reIdEmbedding`, gate on `cosineSimilarity(lockedReIdEmbedding, candidate.reIdEmbedding)` instead of MobileNetV3. Use MobileNetV3 as a secondary signal but not as a gate. Keep the existing label-override / geometric-override paths; just compute "appearance score" from OSNet for persons and from MobileNetV3 for everything else.

**Expected impact on failure cases**:
- Two-people: large. OSNet was trained to discriminate persons across cameras. Cosine of 0.6+ between same-person across viewpoints is the typical regime.
- Two-vehicles: zero (OSNet is person-only).

**Cost**: <1 day. The plumbing is all there: `lockedReIdEmbedding`, `candidate.reIdEmbedding`, async computation of OSNet for top-2 candidates. The only changes are in `scoreCandidate`:
1. Introduce a person-specific gate that uses OSNet sim if available.
2. Make sure `addSceneNegative` doesn't poison the MobileNetV3 negatives based on a candidate that OSNet thinks is the locked person.

**Risk**:
- OSNet is currently **only computed for the top-2 same-label candidates** by MobileNetV3 sim (ObjectTracker line 959–967). If MobileNetV3 sim ranks the wrong candidate higher among persons, OSNet won't be computed for the right one. Need to also compute OSNet for *every* person candidate during search, not just top-2 by generic sim. This adds cost: ~5–10ms per OSNet inference × ~3 person candidates = +15–30ms per search frame. Acceptable; we're already running detector + embedder.
- OSNet was trained on Market-1501 (surveillance crops, full-body). Phone footage has aggressive close-ups and partial bodies. Worth a model-quality test in `tools/` before shipping.

**Verification**:
- Capture 2-3 two-people scenarios and add as scenario JSON. In each, the scoring engine should reacquire the correct person within `maxFramesLost / 2` frames after LOST.
- Add unit test: `ReacquisitionEngine` with `lockedReIdEmbedding != null`, two person candidates with different `reIdEmbedding` values (one near the lock, one far), confirm the near one wins regardless of MobileNetV3 sim.
- The benchmark in `tools/benchmark_embeddings.py` should be extended to score OSNet embeddings on real person crops.

### C.2 Stop self-poisoning the negative pool during search

**Summary**: Today, every rejected candidate gets `addNegativeExample()`'d (lines 526, 582). When the floor is misconfigured (failure case), the locked person gets repeatedly added as a negative — the worst possible thing. Change to: only add scene negatives during *confirmed* tracking (already done by `addSceneNegative`'s 0.85 filter), or add only when the geometric override **would** have accepted the candidate but the embedding sim is below floor (i.e. someone clearly different), or just don't add negatives during search at all.

**Expected impact**: prevents the negative pool from learning the locked person is "not us." Mostly relevant for the online classifier and prototype margin signals — those signals are then trustworthy when they fire.

**Cost**: <1 day. One-line change + a unit test that confirms a same-person candidate scoring below floor is NOT added as a negative.

**Risk**: very low. Today's behavior is already mostly broken; not adding negatives at all is strictly better for the failure case. There's a marginal regression where genuinely-different objects' negatives won't be collected during search — but they're collected during confirmed tracking just fine.

**Verification**: in the live person session log, confirm no `Scene negative added` lines appear during search. Unit test: 5 below-floor person candidates, gallery negatives count stays 0.

### C.3 Per-lock learned threshold (replacing fixed `MIN_EMBEDDING_SIMILARITY = 0.15` and adaptive floor 0.3..0.5)

**Summary**: At lock time, compute the spread of cosine similarities the gallery's augmented views have *to each other* and the cosine to a small set of frozen background negatives shipped as an asset. Set the floor as `mu_intra - k * sigma_intra` clamped above the maximum negative cosine. For "rigid object lock with consistent gallery," this floor will be high (e.g. 0.7) — aggressive rejection. For "person lock where augmented views are themselves diverse," this floor will be low (e.g. 0.2) — permissive of within-class variation.

**Expected impact**: directly fixes the "same person is rejected at sim=0.27 because floor is 0.4" failure. The floor is now learned from how variable the embedder is on *this specific lock*.

**Cost**: 1–2 days. Need to ship ~500 background negative MobileNetV3 embeddings as an asset (~3MB), implement the threshold derivation in `ReacquisitionEngine.lock()`.

**Risk**: under-thresholds for visually homogeneous scenes (e.g. someone in a black T-shirt against a black couch — gallery becomes too tight). Mitigate by clamping the floor to `[0.15, 0.55]` and add a unit test.

**Verification**: replay the captured person session through the engine with the new threshold; expect `id=234` to be reacquired within ~30 frames (the sim values 0.20–0.34 should pass an adaptive floor that learned the gallery is loose).

### C.4 Replace MobileNetV3-Large with MobileCLIP-S2 (or DINOv2-small)

**Summary**: MobileNetV3-Large is a 2019 ImageNet-classification model, repurposed for similarity. MobileCLIP-S2 (or DINOv2-small) was actually *trained* for instance-level visual similarity. The replacement should produce stronger same-category discrimination across the board (cars, bottles, mugs, persons-without-OSNet).

**Expected impact**:
- Two-vehicles: large. This is the only lever that meaningfully discriminates "red car A vs red car B" without a vehicle-specific re-ID.
- Two-people: medium (OSNet covers persons better; the generic embedder's job for persons is "whole-body context as fallback").
- Same-category rigid objects (two mugs, two bottles): large.

**Cost**: 3–5 days. The bulk is conversion (PyTorch checkpoint → ONNX → onnx2tf → TFLite + GPU delegate compatibility), benchmarking on the same fixtures we have in `tools/`, and validating GPU-delegate behavior on Adreno 740. We'd run as raw TFLite (not MediaPipe ImageEmbedder), same pattern as `magic_touch.tflite` and OSNet.

**Risk**:
- Conversion can fail or produce a model that's slow on GPU. Benchmark before shipping. Have a CPU fallback already wired (`createGpuInterpreter`).
- MobileCLIP outputs ~512-dim instead of 1280-dim. Gallery storage shrinks (good). The augmented gallery may behave differently — reset thresholds.
- DINOv2 needs 224×224 input via patch-14, which is slightly awkward sizing; MobileCLIP is more flexible.

**Verification**: extend `tools/test_model_quality.py` with the same fixtures (`reacquisition_basic`, `two_apples`) plus a new `two_cars` fixture. The test should assert that **same-instance cosine ≥ 0.6** and **different-instance cosine ≤ 0.4** for the median pair across the fixture, for both apples and cars.

### C.5 K-reciprocal re-ranking on the gate survivors

**Summary**: Among gate survivors, compute the k-reciprocal nearest neighbor relationship between the candidate and the gallery: candidate qualifies if it's in the gallery's top-k AND the gallery has any embedding in the candidate's top-k of-itself-vs-scene. This is essentially free at our gallery sizes.

**Expected impact**: small but meaningful. Mainly improves robustness when one bad gallery embedding (e.g. a back-of-head augmentation) accidentally has a high cosine to a wrong candidate. Reciprocal check filters those out.

**Cost**: <1 day. Pure algorithmic. ~50 lines of Kotlin in `ReacquisitionEngine`.

**Risk**: very low.

**Verification**: replay test that constructs a gallery + two candidates where `bestGallerySimilarity` ranks the wrong one but k-reciprocal ranks the right one. Demonstrate the new code prefers the right one.

### C.6 Online classifier from MobileCLIP + frozen-asset negatives (Exemplar-SVM)

**Summary**: The existing `OnlineClassifier` requires runtime negatives, which the failure case can't supply. Ship 500–2000 pre-computed embeddings of generic background (random ImageNet/ADE20K crops) as a TFLite asset, ~1–4MB. At lock time, train the classifier on `gallery (positive)` + `frozen_negatives (negative)` immediately. Gate on classifier output.

**Expected impact**: solves the "no negatives, no learned boundary" cold-start problem the current pipeline has during the first ~30 frames of search. Combined with MobileCLIP, this is essentially production-ready instance retrieval.

**Cost**: 1–2 days. Generate negative embeddings in `tools/`, ship as asset, change `lock()` to immediately train the classifier with frozen + scene negatives (vs only when both ≥ 3).

**Risk**: low. Works strictly better than the current cold-start (which is "no classifier"). The MobileNetV3 frozen negatives can be generated once and shipped.

**Verification**: same as C.3.

### C.7 Smarter sync-fallback: embed all persons during search, not just closest

**Summary**: Currently `withSyncFallback` (ObjectTracker line 662) computes a sync MobileNetV3 embedding for the *single closest* same-category candidate. With two persons in scene, this systematically embeds the wrong one. Compute sync embeddings for the top-2 same-category candidates, or for all same-category candidates at the cost of ~2×8ms per frame.

**Expected impact**: fixes a specific class of failures where the wrong person sits closer to the last-known position. Without this, even C.1 (OSNet gate) won't help because OSNet only runs after MobileNetV3 has at least produced an embedding for the candidate.

**Cost**: <1 day.

**Risk**: extra compute per search frame (maybe +16ms). Could be reduced to top-2 to bound cost.

**Verification**: replay test with two-person scene; both candidates get embeddings; correct one selected.

### C.8 Diverse gallery construction (multi-view, "wait for movement")

**Summary**: Today's gallery starts with 5 augmentations (3 rotations + flip + original). Rotations and flips don't actually represent multi-view appearance — they distort the original view. What we actually want is the same person from front, side, back. This requires *time*: either don't fully commit to lock until VT has run for 15–30 confirmed frames during which we see the person in 2–3 different poses, or always accumulate the gallery from confirmed VT frames at moderate IoU change (=different angle).

**Expected impact**: medium. Fixes specifically the "lock on a person facing forward, lose them, they reappear sideways" failure. Already partially addressed by accumulating during VT (line 421), but the cap and diversity check are too strict for persons.

**Cost**: 1–2 days. Mostly tuning + scenario validation.

**Risk**: longer time-to-lock perception for users. Might want to make this an opt-in for "high-stakes lock" or always lock immediately but flag the gallery as "single-view" until diversity arrives.

**Verification**: visual scenario test with a person who turns 90° during VT.

### C.9 Test-time augmentation (TTA) at gate time for borderline candidates

**Summary**: When a candidate's MobileNetV3 sim is in `[floor - 0.05, floor + 0.05]` (uncertainty zone), embed three slight variants of the candidate crop (small jitter, horizontal flip, slight scale), take the max sim. Cheap way to recover ~2-5% of borderline rejections without model changes.

**Expected impact**: small. Specifically helps the "sim 0.36, floor 0.40" near-misses that are common in the chair regression we mentioned.

**Cost**: <1 day.

**Risk**: extra compute on borderline cases. Cap at ~3 augmentations to bound cost.

**Verification**: replay session where chair sim varies 0.40–0.75; expect the 0.36–0.40 borderline frames to now pass.

### C.10 Body-part attention via mask × embedding pooling

**Summary**: Today the embedder pools features over the full crop. With our segmentation mask, we could pool only over the foreground pixels (essentially what we do now via masking) but additionally weight upper-body and lower-body separately to emphasize clothing color over background-similar features. PCB-style striping but at the embedding pooling layer rather than the network architecture.

**Expected impact**: small to medium for persons. Probably not the right battle right now (compose with MobileCLIP+OSNet-gate first).

**Cost**: 3–5 days, requires modifying the embedder output post-process.

**Risk**: medium — needs careful tuning, the gain over OSNet (which already does roughly this internally) is unclear.

**Verification**: same fixtures as C.4.

---

## D. Recommended sequence

The next 2–3 PRs should be:

### PR1 — "Persons gated on OSNet, generic embedder demoted to ranking signal"

**Hypothesis**: Two-people failures are caused by gating on a generic ImageNet embedder. If we gate on OSNet (which we already compute) for person-locked sessions, two-people reacquisition becomes reliable.

**Ship**:
1. In `scoreCandidate`, when `lockedIsPerson && candidate.label == "person"`, derive `appearanceScore` from `cosineSimilarity(lockedReIdEmbedding, candidate.reIdEmbedding)` if available, fall back to MobileNetV3 if not.
2. In `ObjectTracker.computeEmbeddingsSync`, compute OSNet for **all** person candidates during search (not just top-2 by MobileNetV3).
3. In `ReacquisitionEngine.scoreCandidate`, **stop** calling `addNegativeExample` on rejected candidates during search (move that into VT-confirmed tracking only).
4. Adapt the `MIN_EMBEDDING_SIMILARITY` and adaptive floor for the OSNet path: OSNet cosine on Market-1501 typically lands in `[0.4, 0.85]` for same-person across cameras; floor can be `0.45–0.55`. Validate by replaying the captured session.

**Revert criteria**: if the new gate causes any non-person regression (e.g. cup or chair scenarios reacquire 2× slower), revert. If person reacquisition is no better than current on two-people scenes, revert.

**Test**: capture a fresh two-people indoor scene (bedroom or living room with two adults visible) and a two-people outdoor scene (e.g. park). Add as scenarios. Both should reacquire the correct person within ~60 frames of LOST. Re-run all existing 184 unit tests.

### PR2 — "Per-lock adaptive embedding floor + frozen-negative classifier cold-start"

**Hypothesis**: Even with OSNet gating, MobileNetV3 floors are too coarse for some locks. A floor learned from gallery intra-spread + a frozen scene-negative pool removes the cold-start gap and the false-floor problem.

**Ship**:
1. Pre-compute ~1500 MobileNetV3 embeddings of generic indoor + outdoor background patches; ship as a binary asset (~7MB).
2. In `ReacquisitionEngine.lock()`, immediately train the `OnlineClassifier` against `gallery (pos) + frozen_negatives (neg)`. Engine starts in classifier-active mode from frame 1.
3. Replace `MIN_EMBEDDING_SIMILARITY` with: `floor = max(0.15, mean(galleryToNegativeSims) + 0.5 * std(galleryToNegativeSims))`, clamped to `[0.20, 0.55]`. Same idea on the OSNet path with its own asset (~5MB OSNet negatives).

**Revert criteria**: if this regresses cup/chair reacquire times by >20%, revert. If it doesn't measurably help two-people scenarios on top of PR1, revert.

**Test**: replay the captured two-people scenarios from PR1 plus a captured two-cars outdoor scenario. Expect at least one of:
- two-cars reacquire within `maxFramesLost`
- two-people reacquire ≥1 second faster than after PR1

### PR3 — "Replace generic embedder with MobileCLIP-S2 (or DINOv2-small)"

**Hypothesis**: MobileNetV3-Large's instance discrimination ceiling is too low for non-person same-category cases (cars, bottles). MobileCLIP-S2 has demonstrably stronger instance-level similarity in literature.

**Ship**:
1. Convert MobileCLIP-S2 image encoder to TFLite (PyTorch → ONNX → onnx2tf, run via raw TFLite GPU). Ship `mobileclip_s2_image.tflite` (~50–80MB FP16). Consider also DINOv2-small as fallback if conversion fails.
2. Wire as `AppearanceEmbedder` replacement (the segmenter integration stays unchanged).
3. Re-tune the OSNet+MobileCLIP fusion in `scoreCandidate` (likely OSNet stays primary for persons, MobileCLIP becomes the primary signal for non-persons).
4. Update `tools/benchmark_embeddings.py` to validate same-instance / different-instance separation.

**Revert criteria**: if real-device inference is >25ms (vs ~8ms for MobileNetV3) and this exceeds frame budget, revert and ship DINOv2-small or just keep MobileNetV3. If accuracy gain on `tools/` fixtures is <5%, not worth the APK bloat — revert.

**Test**: re-run ALL captured scenarios. Expected behavior is that nothing regresses and that the two-cars and two-mugs scenarios from PR1/PR2 reacquire reliably (vs probably-not-reacquiring with PR1+PR2 alone). Add a dedicated two-cars fixture to `tools/test_model_quality.py`.

---

## E. Things explicitly NOT recommended

- **Going back to weighted-average scoring** — already deprecated (CLAUDE.md), correctly.
- **Dropping the cascade gates entirely** — they're the right structure; the question is what populates `appearanceScore`.
- **Server-side embedding** — out of scope.
- **Vehicle-specific re-ID model** — too narrow, and SoTA generic embedders cover the case adequately.
- **Re-introducing YOLOv8-oiv7** — won't help identity at all, just label flicker which we no longer gate on.
- **Tuning more thresholds** — the MEMORY note "feedback_no_parameter_tweaking" applies; the structural fix (PR1+PR2+PR3) makes thresholds matter less, not more.

---

## F. Confidence notes

- Highest confidence: **the OSNet-gating fix (PR1) is correct**. Direct read of `scoreCandidate()`'s rejection path, direct read of the live log with 97 same-person rejections, direct read that OSNet was being computed but never consulted. This is a coding gap, not a research gamble.
- Medium confidence: **MobileCLIP-S2 conversion will succeed** on Adreno 740 with reasonable inference latency. The model is designed for mobile but the actual community TFLite conversions vary in quality.
- Lower confidence: **two-cars works after PR3**. I have no captured two-cars session to verify against; the reasoning is by analogy to two-mugs / two-apples in `tools/test_fixtures/`, plus generic claims about MobileCLIP's instance-retrieval performance. Ship a captured two-cars scenario before claiming the win.
- Honest unknown: **the actual OSNet inference cost on Adreno 740** for 3–5 candidates per search frame. The model is small (4.2MB) but TFLite GPU delegate behavior on this Snapdragon is well-known to be unpredictable for non-MediaPipe models. Have a CPU fallback ready.

---

## Appendix: file/line references for the load-bearing claims

- Generic embedder is the only gate: `ReacquisitionEngine.kt` lines 502–528 (appearanceScore from `bestGallerySimilarity` only), lines 525–528 (floor enforcement returns null before re-ID/face are touched).
- OSNet/MobileFaceNet are scoring-only: `ReacquisitionEngine.kt` lines 604–651 (the `hasFace` and `hasReId` ranking branches all run *after* `appearanceScore` cleared the gate).
- OSNet is computed only for top-2 by generic sim during search: `ObjectTracker.kt` lines 959–977.
- Self-poisoning negatives during search: `ReacquisitionEngine.kt` lines 526 and 582 (`addNegativeExample(candidate.embedding!!)` inside the rejection paths).
- Live log evidence: `/tmp/sessions_62_baseline/session_20260424_203704_person/session.log` — 97 lines containing `rejected id=234 label="person" sim=0.xxx (hard threshold)`.
- Cup session for contrast: `/tmp/sessions_62_baseline/session_20260424_204120_cup/session.log` — sim 0.696, 0.836, 0.863, 0.907 in clean reacquires.
- Lock-time gallery construction: `AppearanceEmbedder.kt` lines 113–141; `ObjectTracker.kt` lines 226–268.
- Gallery accumulation logic (and the centroid-vs-self diversity check note): `ObjectTracker.kt` lines 421–432.
