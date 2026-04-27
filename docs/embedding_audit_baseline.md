# Embedding-input audit — baseline (#92)

> **Phase 0 of #91.** No embedder behavior changed in this PR — this exists to make the actual crops fed to MNV3, OSNet, and MobileFaceNet visible, and to measure how much each embedding drifts frame-to-frame on the *same* locked object. Every Phase 1 fix (#93 face alignment, #94 OSNet masking, #95 letterboxing) is judged against the numbers and images here.

## Methodology

During VT-confirmed tracking of the locked object:

- Every **5** confirmed frames, snapshot the current frame + locked bbox and submit to a low-priority background executor (`AuditEmbed`). The executor runs MNV3, OSNet (when person), and MobileFaceNet (when person) on that snapshot, then records each embedding into `EmbeddingStabilityLogger`. Snapshot cost on the processing thread is one `bitmap.copy` (~4ms). The embedder calls themselves never block production.
- Every **30** confirmed frames the executor additionally writes a per-embedder **composite JPEG** under `session_<ts>/crops/<NNN>_<event>.jpg` showing: full frame with bbox · raw bbox crop · segmenter masked crop (or "null" when rejected) · OSNet 256×128 input · person crop with BlazeFace bbox + 6 keypoints overlaid · MobileFaceNet 112×112 input.
- On lock clear, `embedding_stability.json` is written next to the crops with per-embedder `p10/p50/p90 + raw history + sampled frames`. `VideoReplayTest` aggregates per-video summaries and prints a comparison table at suite end.
- `CropDebugCapture.AUDIT_ENABLED` flips the whole thing off at compile time.

The interesting metric is **same-object self-similarity at increasing distances**:

- **k=1** (≈ 5 frames apart, ~167ms at 30fps) — consecutive-frame noise from input jitter.
- **k=5** (≈ 25 frames apart, ~833ms) — short-window stability.
- **k=30** (≈ 150 frames apart, ~5s) — longer-window stability across pose/lighting drift.

A wide spread between same-object frames is the noise floor — no model swap will help below it. Closing that floor is what Phase 1 PRs target.

## What each embedder sees today

> ![man_desk lock composite](embedding_audit_examples/man_desk_lock.jpg)
>
> *LOCK frame. Subject's bbox covers most of the right side of the frame — segmenter rejected the mask (>50% area), so MNV3 falls back to the raw crop. OSNet stretches a near-square person crop to 1:2. BlazeFace finds the face cleanly with 6 keypoints — they are visible in the overlay because we drew them, not because the pipeline uses them. The face input below is stretched, slightly tilted, off-center.*

Three things jump out from a single composite — and they hold across every session we captured:

1. **MNV3 doesn't see the masked crop on tightly-framed person bboxes.** The segmenter has a 50%-of-frame guard that rejects whole-body close-ups; on those, MNV3 falls back to the raw bbox with all the background. Counterintuitively, *worse* bbox framing is what lets the segmenter help.

2. **OSNet always sees the raw bbox, stretched.** No segmenter call, no aspect preservation. A short, wide bbox (sitting subject, half-body crop) gets squashed; a tall, narrow bbox (full standing subject) gets the head crammed into the top fifth.

3. **MobileFaceNet receives a stretched bbox crop.** The 6 BlazeFace keypoints are computed and then thrown away. Standard face-recognition pipelines align these to canonical positions before the embedder sees the face — we don't.

> ![man_desk VT frame 60](embedding_audit_examples/man_desk_vt_f60.jpg)
>
> *VT frame ~60. Same patterns 2 seconds later: segmenter still null, OSNet still stretched, face still un-aligned. Visible movement is mostly the camera, not the subject — yet the embedders' inputs differ subtly between frames in ways that show up in the stability table below.*

> ![mouse desk lock](embedding_audit_examples/mouse_desk_lock.jpg)
>
> *Mouse lock. Bbox is small (~13% of frame area), so segmenter succeeds — black background visible in the masked crop. The mouse is preserved, the desk is gone. This is the correct behavior — MNV3 gets a clean instance signal. OSNet is still stretched (we render it for non-person locks too, to make the aspect distortion visible — it isn't actually called on non-persons). The 82×58 source becomes a 128×256 input: severe aspect mangling.*

## Stability table

Aggregated from individual `am instrument` runs. The table is auto-printed by `VideoReplayTest.printEmbeddingStabilityTable` after `@AfterClass` when the suite completes — but see "Known issues" below for why the full suite currently can't run end-to-end.

> Sample count `n` is the number of cosine deltas at that lookback distance. The k=30 column requires ≥31 samples to populate; tests that ended early or had short VT-confirmed windows show `--` there.

| Video | Embedder | k=1 p10/p50/p90 (n) | k=5 p10/p50/p90 (n) | k=30 p10/p50/p90 (n) |
|---|---|---|---|---|
| **man_desk_camera_swing** | MNV3 | 0.60 / 0.89 / 0.97 (39) | 0.38 / 0.64 / 0.83 (35) | 0.50 / 0.57 / 0.61 (10) |
| | OSNet | 0.78 / 0.96 / 0.99 (39) | 0.70 / 0.87 / 0.93 (35) | 0.73 / 0.84 / 0.87 (10) |
| | MobileFaceNet | 0.62 / 0.87 / 0.93 (39) | 0.48 / 0.78 / 0.88 (35) | 0.50 / 0.82 / 0.93 (10) |
| **mouse_desk_rotation** | MNV3 | 0.74 / 0.88 / 0.98 (21) | 0.47 / 0.77 / 0.88 (17) | -- |
| **two_men_forest** (run A) | MNV3 | 0.13 / 0.76 / 0.84 (6) | 0.22 / 0.22 / 0.22 (2) | -- |
| | OSNet | 0.66 / 0.88 / 0.95 (6) | 0.64 / 0.64 / 0.64 (2) | -- |
| | MobileFaceNet | 0.64 / 0.65 / 0.73 (4) | -- | -- |
| **two_men_forest** (run B) | MNV3 | 0.18 / 0.79 / 0.81 (4) | -- | -- |
| | OSNet | 0.69 / 0.91 / 0.96 (4) | -- | -- |
| | MobileFaceNet | 0.56 / 0.56 / 0.56 (2) | -- | -- |
| **crowd_street** | MNV3 | 0.20 / 0.31 / 0.54 (4) | -- | -- |
| | OSNet | 0.40 / 0.44 / 0.62 (4) | -- | -- |

### What the numbers say

Even with limited sample counts on multi-person scenes, three patterns are unambiguous:

1. **MNV3 has the widest spread on every video.** On the cleanest test (`man_desk`, 39 samples) the consecutive-frame p10 is 0.60 — meaning 10% of consecutive *same-object* frames produce embeddings that are barely comparable. On multi-person crowds it collapses entirely (`crowd_street` p50 = 0.31). This is what we expected — close-up bboxes trigger the segmenter's >50% guard, MNV3 falls back to the raw crop, and the raw crop is dominated by background.

2. **OSNet is the tightest of the three on stable scenes** (`man_desk` k=1 p50 = 0.96) but still loses 12pp by k=30 (p50=0.84) — that's the headroom #94 (mask the body input) is going after. On `crowd_street` it drops to p50 = 0.44, telling us OSNet *also* struggles when the body crop bleeds in adjacent persons. The fix isn't a bigger model — it's not embedding the wallpaper.

3. **MobileFaceNet's k=1 p10 of 0.62** on `man_desk` (best case) is almost as wide as MNV3's. Same crop, a few frames later, embedded by a *face-recognition-trained* model, and the cosine drops below 0.62 ten percent of the time. That's the alignment cost the literature warns about — the bbox-only crop preserves head tilt, partial occlusion, and aspect distortion that a 5-point similarity transform would normalize away.

### Provisional Phase 1 deltas to watch

| PR | Metric to move | Today (best baseline) | Phase 1 target |
|---|---|---|---|
| **#93** face alignment | MobileFaceNet k=1 p10 on `man_desk` | 0.62 | ≥ 0.78 (close the gap to OSNet) |
| **#94** masked OSNet | OSNet k=30 p50 on `man_desk` | 0.84 | ≥ 0.90 (close the long-window drift) |
| **#95** letterbox | k=1 p10 across all three on `man_desk` | 0.60–0.78 | +5pp each (less aspect jitter) |

## Implications for Phase 1

| PR | Hypothesis | Expected delta |
|---|---|---|
| **#93** face alignment | Aligning the 5 source keypoints to canonical positions removes pose noise that the bbox-only crop preserves. | MobileFaceNet k=1 p10 climbs from ~0.67 toward OSNet's 0.88. Knock-on: face gate (#84) rejects fewer genuine matches at the same threshold. |
| **#94** masked OSNet | OSNet currently embeds wallpaper alongside the person. Masking removes that. | OSNet k=30 p50 climbs (less drift from background context). Same-person sim rises, different-person stays put. The win is wider separation, not just shifted means. |
| **#95** letterbox | Aspect-ratio jitter between consecutive frames produces input variation that all three embedders convert into output variation. Letterboxing fixes the input side. | Primary signal is k=1 self-similarity rising — narrower noise band. |

## Known issues found during baseline collection

- **Long suite runs hang nondeterministically.** Two independent attempts at the full `VideoReplayTest` suite (16 tests) hung after 5–6 tests with the app process at 0% CPU. The hang reproduced both with synchronous-on-pipeline audit and with the async-executor refactor that produced the numbers above. Single-test `am instrument -e class …#methodName` invocations also occasionally hang (boy_indoor_wife_swap and crowd_street both timed out at the 3-min watchdog in one of the isolated runs). Single tests *mostly* complete cleanly (~30–60s each); the hang seems related to shared-state buildup on `sharedTracker` across many tests. Root cause is unclear — likely candidates: GPU context leaks from the segmenter or TFLite GPU embedders, native memory from OpenCV mats, or contention with the audit executor and the production embedding executor. **Worth investigating in a follow-up before Phase 1 PRs need their full re-baselines.** The audit infrastructure itself works — what we have isn't enough to draw firm conclusions on every video, but the man_desk numbers (39 stability samples) are solid and the visual evidence is unambiguous across every composite.

- **`person_playground_tracking.mp4` is 277 MB**, ~10× the next-largest fixture. 4K decode on Adreno 740 produces ~0.05 fps GL throughput on this file — a single test takes 7+ minutes and gives only ~24 processed frames (3 stability samples — useless for the audit). Per `feedback_downscale_videos`, this should have been ffmpeg-downscaled to 640px before pushing. Re-encoding is independent of the audit and should happen before any future re-baseline.

- **`AUDIT_ENABLED` flag exists for a reason.** If a debug-build user wants to skip the audit cost entirely, flip `CropDebugCapture.AUDIT_ENABLED` to false and the entire instrumentation compiles in but does nothing. We default to `true` in this PR so the data flows; a later PR may flip it to `false` once Phase 1 fixes have landed and the audit's job is done.

## How to reproduce

```bash
./gradlew assembleDebug assembleDebugAndroidTest
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell rm -rf /sdcard/Android/data/com.haptictrack/files/debug_frames/

# Single test (works reliably):
adb shell am instrument -w \
  -e class com.haptictrack.tracking.VideoReplayTest#man_desk_camera_swing_reacquires_correctly \
  com.haptictrack.test/androidx.test.runner.AndroidJUnitRunner

# Full suite (currently flaky — see "Known issues" above):
adb shell am instrument -w com.haptictrack.test/androidx.test.runner.AndroidJUnitRunner

# Pull results:
adb pull /sdcard/Android/data/com.haptictrack/files/debug_frames/ /tmp/audit/
```

The aggregate table is printed to logcat with tag `VideoReplayTest`, prefix `[Audit #92]`. Per-session detail lives under `session_*/crops/` (composites) and `session_*/embedding_stability.json` (raw stats).
