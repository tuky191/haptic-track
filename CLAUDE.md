# HapticTrack

Hands-free camera tracking Android app. Uses on-device object detection + haptic feedback + auto-zoom to let you film a subject without looking at the screen.

## Development Workflow

All changes follow this flow — never commit directly to master:

1. **Branch** — create a feature/fix branch from master
2. **PR** — push and open a pull request
3. **Device test** — build, install via ADB, test on physical device
4. **Capture scenario** — every device test should produce a `scenario.json` for replay testing
5. **Review** — review the PR (code review or self-review)
6. **Merge** — merge to master and delete the branch

## Quick Reference

```bash
./gradlew assembleDebug          # Build debug APK
./gradlew testDebugUnitTest      # Run unit tests (~3s, 301 tests)
adb install -r app/build/outputs/apk/debug/app-debug.apk  # Deploy to device

adb logcat -d -s "Reacq" -s "VisualTracker" -s "FTFTracker"  # Pull tracking logs
adb logcat -c                                                                 # Clear logcat
adb pull /sdcard/Android/data/com.haptictrack/files/debug_frames/             # Pull debug overlays + scenario.json
adb shell rm -f /sdcard/Android/data/com.haptictrack/files/debug_frames/*.png # Clear debug frames

# Python model quality tests
cd tools && .venv/bin/pytest test_model_quality.py -v
cd tools && .venv/bin/python benchmark_embeddings.py test_fixtures/two_apples/scenario.json /path/to/frames/
```

## Source Layout

```
app/src/main/java/com/haptictrack/
├── MainActivity.kt, HapticTrackApp.kt     # Compose entry point
├── camera/                                # CameraManager, DeviceOrientationListener
├── tracking/                              # ← most of the complexity lives here
│   ├── ObjectTracker.kt                   # Orchestrator
│   ├── ReacquisitionEngine.kt             # Cascade scoring (gates + z-norm + ROSTER_REJECT)
│   ├── SessionRoster.kt                   # Per-session non-lock person memory (#113)
│   ├── AppearanceEmbedder.kt              # MobileNetV3 generic embedding + gallery
│   ├── FaceEmbedder.kt, PersonReIdEmbedder.kt   # Person path (face + body)
│   ├── PersonAttributeClassifier.kt       # BlazeFace + age-gender (Crossroad-0230 removed in #124)
│   ├── VisualTracker.kt                   # OpenCV VitTracker wrapper
│   ├── FrameToFrameTracker.kt             # IoU-based detection ID assignment
│   ├── DetectionFilter.kt, ObjectSegmenter.kt, ScenarioRecorder.kt, DebugFrameCapture.kt
│   ├── CanonicalCrop.kt, CropClassifier.kt, FrozenNegatives.kt
│   ├── VelocityEstimator.kt, EmbeddingStabilityLogger.kt, CropDebugCapture.kt
│   ├── EmbeddingUtils.kt                  # Shared helpers (IoU, cosine, histogram, z-norm)
│   └── TrackingState.kt                   # Data classes
├── ui/                                    # CameraScreen, CameraViewModel
├── haptics/HapticFeedbackManager.kt
└── zoom/ZoomController.kt

app/src/test/      # Robolectric unit tests
app/src/androidTest/  # On-device instrumentation (VideoReplayTest, OsnetGrayBiasTest)
```

## Architecture

MVVM with a single-activity Compose UI. Two-tier tracking: visual tracker for frame-to-frame, EfficientDet-Lite2 detection for re-acquisition.

```
CameraViewModel
├── CameraManager              (CameraX: SurfaceTexture GL pipeline, zoom, video capture)
├── DeviceOrientationListener  (accelerometer: detects upside-down/landscape holds)
├── ObjectTracker              (orchestrates the tracking pipeline below)
│   ├── VisualTracker          (OpenCV VitTracker: primary frame-to-frame pixel tracking)
│   ├── MediaPipe Detector     (EfficientDet-Lite2: object detection, 80 COCO classes, every frame)
│   ├── FrameToFrameTracker    (IoU-based ID assignment for detections)
│   ├── AppearanceEmbedder     (MobileNetV3: generic visual identity fingerprint)
│   ├── FaceEmbedder           (MobileFaceNet: face identity for person path)
│   ├── PersonReIdEmbedder     (OSNet x1.0 MSMT17: body re-ID for person path)
│   ├── PersonAttributeClassifier (BlazeFace + age-gender: person attributes)
│   ├── SessionRoster          (per-session non-lock person memory → ROSTER_REJECT margin gate)
│   ├── ReacquisitionEngine    (cascade scoring: hard category gate + z-norm calibration + ranking)
│   ├── ScenarioRecorder       (captures processFrame inputs as JSON for replay testing)
│   ├── DetectionFilter        (noise removal)
│   └── DebugFrameCapture      (saves annotated frames + session logs on tracking events)
├── HapticFeedbackManager      (vibration patterns mapped to tracking status)
└── ZoomController             (auto-zoom from bounding box size/position)
```

### Data Flow

**Normal tracking (object visible):**
1. CameraX feeds frames to `ObjectTracker` via the SurfaceTexture GL pipeline (always active, regardless of recording state)
2. `DeviceOrientationListener` provides physical rotation → bitmap rotated to upright
3. `VisualTracker` (VitTracker) updates the locked object's position by pixel correlation
4. Detector runs in parallel to cross-check — if detector confirms the label at the tracker's box, tracking continues
5. Every 5 frames, the current VT crop is embedded and compared against the lock gallery — primary drift signal (catches drift even when the detector can't see the target). The mismatch threshold (`templateLow`) is per-lock, derived from gallery self-recall (`lockSelfFloor × 0.75`)
6. 3 consecutive template mismatches OR 10 frames without detector confirmation → drift detected. If a same-category detector box covers VT but the VT box has both grown ≥1.3× since lock and is ≥1.3× the detector box, VT is reseated on the smallest matched detection instead of killed (handles VT runaway during legitimate zoom)

**Re-acquisition (object lost):**
1. Detector runs, `FrameToFrameTracker` assigns stable IDs via IoU matching
2. `AppearanceEmbedder` computes generic embeddings; for person candidates, `FaceEmbedder` and `PersonReIdEmbedder` add face + body re-ID embeddings (async pipeline; synchronous fallback for the closest same-category candidate when the async cache can't bridge across frames during rotation)
3. `DetectionFilter` runs before scoring to strip truly full-frame detections (area > 0.85) — tentative confirmation handles flickering phantoms structurally
4. `ReacquisitionEngine` applies a **hard person/not-person category gate** (no override — cross-category candidates are rejected outright), then ranks survivors using z-score-calibrated embedding similarities (Phase 4: raw cosine is mapped through a sigmoid against the live impostor distribution once the cohort matures)
5. `SessionRoster` checks each non-lock person slot — if a non-lock body+face score beats the lock by `ROSTER_REJECT_MARGIN`, the candidate is rejected as "more likely a known impostor"
6. Strong embedding match (raw > 0.7 or z-score > 1.0) overrides position/size hard thresholds (geometric override only; never the category gate)
7. Best candidate above `minScoreThreshold` becomes the new lock; visual tracker re-initializes
8. `ScenarioRecorder` captures every frame's detections and events as JSON for replay testing

**Recording (4K mode):**
1. Recording toggles VideoCapture on/off — no rebind or mode switching needed
2. Frames always come from the SurfaceTexture GL pipeline, same as normal tracking
3. 4K recording is always available since the pipeline uses only 2 streams (preview + video)
4. Tap-to-lock auto-starts recording (PR #111) — the first lock in a session begins capture so users don't miss the moment

**Stealth mode:**
- Black overlay covers the preview — purely a UI layer (opaque black Box in Compose)
- SurfaceTexture pipeline provides frames regardless of UI visibility
- Volume-down cycle still works: lock → record → stop+clear

**Display:**
1. `DetectionFilter` removes noise before sending to UI
2. `CameraViewModel` updates `TrackingUiState` (StateFlow), drives haptics + zoom
3. `CameraScreen` renders bounding boxes with FILL_CENTER coordinate transform

## Scenario Replay Testing

`ScenarioRecorder` captures every `processFrame` input (detections, embeddings, color histograms, attributes) as `scenario.json` in the session's debug directory. Any failure can be replayed deterministically as a unit test:

1. Reproduce on device → `adb pull .../debug_frames/session_*/scenario.json`
2. Copy to `app/src/test/resources/scenarios/`
3. Add a `@Test` in `ScenarioReplayTest` using `loadScenario(...)` + `replay(...)` + `assertReacquiresLabel(...)` / `assertNeverReacquiresLabel(...)` / `assertTimesOut(...)` / `buildSyntheticScenario(...)`
4. `./gradlew testDebugUnitTest`

FloatArrays are base64-encoded little-endian IEEE 754. Serialization helpers + `replay()` + asserts live in `ScenarioRecorder.kt` and `ScenarioReplayTest.kt`. JSON schema is documented inline in `ScenarioRecorder.kt`.

## Re-acquisition Scoring (cascade gates)

`ReacquisitionEngine.scoreCandidate()` uses DeepSORT-style cascade gates instead of a weighted average. Wrong-category candidates are rejected outright — no amount of color or position similarity can rescue them.

### Gate A: Hard person/not-person category gate

The first check, with **no override**. If the locked object is a person and the candidate isn't (or vice versa), the candidate is rejected unconditionally — even a perfect embedding match cannot pass. This replaced the old soft `labelOverride` (PR #119) and structurally eliminates cross-category leaks like the kid-to-wife / boy-to-chair cases.

### Gate B: Geometric hard filters

Candidates are rejected if they fail position/size checks (unless embedding override):

- **Position**: center distance > `effectivePositionThreshold` → reject (threshold expands from 0.25 to 1.5 over 30 frames)
- **Size**: size ratio > `effectiveSizeRatioThreshold` → reject (threshold expands over time)
- **Override**: if raw embedding similarity > `APPEARANCE_OVERRIDE_THRESHOLD` (0.7) **or** the z-score exceeds `Z_GEOMETRIC_OVERRIDE_THRESHOLD` (1.0), both position and size filters are bypassed

### Gate C: Within-category label gate

Within the same category, EfficientDet labels still flicker (chair↔couch, bowl↔potted plant). The label gate handles this softly:

| Condition | Result |
|---|---|
| `lockedLabel == null` | Pass (no label constraint) |
| Label matches `lockedLabel` or `lockedCocoLabel` | Pass |
| Raw embedding similarity > 0.7 | Pass (label flicker override) |
| Otherwise | **Reject** |

### Gate D: SessionRoster reject

`SessionRoster` (PR #113) maintains per-session memory of non-lock persons. If a candidate's combined face+body score against a known non-lock slot exceeds `ROSTER_REJECT_FLOOR` (0.55) **and** beats the lock by `ROSTER_REJECT_MARGIN` (0.07), the candidate is rejected — the model thinks this is a known impostor, not the locked person.

### Ranking (among gate survivors): Phase 4 z-norm

Embedding similarities are **calibrated to z-scores** against the live impostor distribution before scoring (PR #119). The cascade ranks on z-mapped scores, not raw cosine:

```
z = (rawCosine - mean) / max(stddev, Z_SIGMA_FLOOR)
calibratedScore = sigmoid((z - 1) × 1)   // Z_SCORE_MIDPOINT=1.0, Z_SCORE_SCALE=1.0
```

| z | calibratedScore | meaning |
|---|---|---|
| -1 | 0.12 | weak impostor |
| 0 | 0.27 | impostor mean |
| 1 | 0.50 | boundary |
| 2 | 0.73 | typical same-person |
| 3 | 0.88 | strong same-person |

z-norm activates per-modality once that modality's cohort has ≥ `MIN_COHORT_FOR_ZNORM` (5) negative samples. Below the threshold, the cascade falls back to raw cosine — typical for single-person scenes that never accumulate enough impostors.

**With z-norm active (multi-person scenes):**
- `hasFace` and `hasReId` person paths use `calibratedFromZ()` for face + body re-ID
- `hasAppearance` (non-person) path stays on raw cosine — the Phase 2 calibration was person-vs-person, the non-person cohort isn't calibrated
- Position (decays), size, color histogram, person attributes contribute at smaller weights
- Label-bonus (+0.05) still rewards exact label matches over the embedding-override pass

**Without embeddings (fallback):** Position (50%, decays) + size (25%) + base bonus (25%). All survivors already passed the gates above.

### Where the code lives

- `ReacquisitionEngine.scoreCandidate()` — cascade gates + ranking (~line 950)
- `ReacquisitionEngine.findBestCandidate()` — scoring loop, picks highest-ranking survivor
- `ReacquisitionEngine.processFrame()` — top-level: direct match → lost → search → reacquire
- `znormToScore()` / `calibratedFromZ()` — sigmoid mapping (~line 430)
- `SessionRoster` — non-lock person memory and roster-reject gate

## Current Work

For PR history: `git log --oneline --first-parent -25`. For open issues: `gh issue list --state open`.

## Key Design Decisions

### Cascade scoring with hard category gate + Phase 4 z-norm
Replaced the original 6-signal weighted average with DeepSORT-style cascade gates (PR #31, #119). Wrong-category (person/not-person) candidates are hard-rejected with **no override** — even a perfect embedding cannot pass. Within-category label flicker is handled softly: a strong embedding (>0.7) bypasses the label gate. Re-id similarities are calibrated per-modality against the live impostor distribution (`raw cosine → z-score → sigmoid`, `MIN_COHORT_FOR_ZNORM=5`) before ranking. `hasAppearance` (non-person) stays on raw cosine — Phase 2 calibration was person-vs-person.

### SessionRoster + scene face-body memory (PR #113, #89)
Per-session memory of non-lock persons (paired face + body embeddings). If a candidate's slot score beats the lock by `ROSTER_REJECT_MARGIN` (0.07) above `ROSTER_REJECT_FLOOR` (0.55), it's rejected as a known impostor. Pairing means asymmetric veto fires when only one modality is present. Closed #108 — the structural answer when no fixed cosine threshold separates same/different person on Adreno 740.

### Adaptive `lockSelfFloor` + dynamic templates (PR #120)
Per-lock floor: `MIN(pairwise self-recall in lock gallery) × 0.7`, clamped `[0.25, 0.40]`. Replaces the fixed `GALLERY_ADD_LOCK_SIM_FLOOR=0.5` from PR #103, which sat above weak-embedder classes' (chair, mouse) cosine band so the accumulator never grew. Same value drives `templateOk` and `templateLow = templateOk × 0.75` for drift detection.

### Detector-anchor VT reseat (PR #120)
VitTracker's regression head lets boxes grow even when tracking is correct. When VT has grown ≥1.3× since lock **and** is ≥1.3× the matched detector box, VT is reinitialized on the smallest matched same-category detection instead of killed. Two conditions guard against legitimate-zoom false-positives.

### Drift detection: template self-verification primary
Every 5 frames during VT tracking, the current VT crop is embedded and compared to the lock gallery. 3 consecutive low-sim checks → drift (primary signal). Detector cross-check (10 unconfirmed frames) is a secondary safety net. Detector-independent so it catches drift even when the detector misses the target. Research basis: TLD (PAMI 2012), Stark (ICCV 2021), DaSiamRPN (ECCV 2018) all treat self-assessment as the primary signal.

### Embedding gallery: augmented + accumulated
At lock, `AppearanceEmbedder` generates 5 embeddings (original + rot 90/180/270 + flip) for immediate multi-angle coverage. During confirmed VT, a new embedding is captured every ~1s, gated on `centroidSim < 0.92 AND lockSim > lockSelfFloor`. Cap `MAX_GALLERY_SIZE=12`.

### Tap-to-track auto-recording (PR #111)
First lock in a session auto-starts recording so the moment isn't lost. Volume-down still cycles idle → lock → start → stop+clear; tap collapses lock+start into one gesture.

### Single-stage detection: EfficientDet-Lite2 (simplified 2026-04-24)
EfficientDet-Lite2 (COCO 80) every frame. YOLOv8n-oiv7 label enrichment was removed once the re-acquisition gate became binary person/not-person — finer OIV7 labels stopped influencing any gate. Saved ~270ms at lock, ~1-2s at startup, 6.8MB APK, one GPU interpreter. `PersonAttributeClassifier` covers person attributes (gender, age) where OIV7 labels were too coarse anyway.

### Unified SurfaceTexture pipeline (PR #54)
Always 2-stream (SurfaceTexture + VideoCapture). Recording toggles VideoCapture on/off — no rebind, no `prepareForRebind()`. Replaced the old 2/3-stream switching where ImageAnalysis was dropped during recording. Always 4K-capable.

### Stealth mode + volume-down cycle
Opaque black Compose `Box` overlays the SurfaceTexture preview — purely UI. SurfaceTexture keeps producing frames regardless of visibility. Volume-down: idle → lock center → start recording → stop+clear.

### GPU delegate for all models
All 11 models run on GPU. EfficientDet-Lite2 is FP16 (was INT8 CPU) — fewer spurious detections. TFLite models use `createGpuInterpreter()` with `Throwable` catch for CPU fallback. MediaPipe models use `Delegate.GPU` via `BaseOptions`. InteractiveSegmenter GPU is capped — Adreno 740 returns empty masks above ~100K pixels, so segment crops are downscaled to ~300x300 (`MAX_SEGMENT_PIXELS = 90000`).

### Adaptive frame skipping during VT
Two-tier. Base (confidence >0.6, confirmed ≥5): every 2nd frame (50% skip). Stable (confidence >0.7, confirmed ≥10): every 3rd frame (67% skip). VT alone is ~5ms vs ~35ms for the detector. Resets on drift, lock clear, or VT stop.

## Logging

### Logcat tags

| Tag | What it logs |
|---|---|
| `Reacq` | LOCK (with `lockSelfFloor=...`), LOST, SEARCH (candidate scores, raw + z-mapped sims), REACQUIRE (with hop count + sim), TIMEOUT, CLEAR, GIVE_UP, OVERRIDE (geometric override fires), ROSTER_REJECT (non-lock slot beat lock), CATEGORY_REJECT (hard person/not-person gate) |
| `VisualTracker` | INIT (tracker started), LOST (confidence dropped), DRIFT (template mismatch OR unconfirmed frames exceeded), VT_RESEAT (detector-anchor reinit), Template mismatch (per-check log when sim below threshold) |
| `SessionRoster` | Slot creation, paired face/body memory updates, asymmetric vetoes |
| `FTFTracker` | NEW (fresh ID assigned), MATCH (IoU match to previous frame) |
| `ScenarioRec` | Recording started/stopped, frame count |
| `AppearEmbed` | Embedding failures, gallery additions |
| `FaceEmbed` / `ReidEmbed` | Per-modality embedding failures |
| `EmbedSync` | Synchronous embedding fallback fired for a candidate that missed the async cache |
| `TFLiteGPU` | GPU delegate activation or CPU fallback per model |
| `DebugCapture` | Saved debug frame filenames |

Filter: `adb logcat -s "Reacq" -s "VisualTracker" -s "FTFTracker"`

### Debug frame capture

`/sdcard/Android/data/com.haptictrack/files/debug_frames/session_<ts>_<label>/` contains annotated + raw PNGs per event (LOCK, LOST, SEARCH, REACQUIRE, TIMEOUT), `session.log`, and `scenario.json`. Auto-prunes to 10 sessions.

## Test Suite

~300 Robolectric unit tests under `app/src/test/`. Hot files: `ReacquisitionEngineTest` (cascade gates, z-norm, lockSelfFloor), `ScenarioReplayTest` (real captured + synthetic scenarios with quantitative thresholds), `SessionRosterTest`, `ZNormTest`, `CanonicalCropperTest`. Everything else is small and named after the class under test.

On-device instrumentation tests under `app/src/androidTest/`: `VideoReplayTest` (mp4 → GL pipeline → tracking-rate thresholds), `OsnetGrayBiasTest` (#102 noise floor harness).

### Python model quality tests

Off-device testing pipeline in `tools/` that validates embedding model quality against real video footage using MediaPipe Python (same engine as Android):

```
tools/
├── requirements.txt              # mediapipe, numpy, Pillow, pytest (Python 3.13)
├── extract_frames.py             # ffmpeg: video → frame PNGs
├── annotate_scenario.py          # OpenCV GUI: draw boxes, assign object IDs
├── benchmark_embeddings.py       # Run model on crops, print similarity matrix + pass/fail
├── test_model_quality.py         # pytest: automated regression checks
└── test_fixtures/
    ├── reacquisition_basic/      # Single apple: lock, pan away, return
    └── two_apples/               # Two red apples + red toy car: identity discrimination
```

Setup: `cd tools && python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## Shared utilities

`EmbeddingUtils.kt` holds top-level helpers reused across tracking classes: `computeIou`, `cosineSimilarity`, `bestGallerySimilarity`, `minPairwiseSimilarity`, `computeCentroid`, `l2Normalize`, `computeColorHistogram`, `histogramCorrelation`, `loadTfliteModel`, `createGpuInterpreter`, base64 ↔ FloatArray.

## Models

| Model | File | Size | Precision | Delegate | Purpose |
|---|---|---|---|---|---|
| EfficientDet-Lite2 | `efficientdet-lite2-fp16.tflite` | 11.6MB | FP16 | GPU (MediaPipe) | Primary detector (80 COCO classes, every frame) |
| MobileNetV3 Large | `mobilenet_v3_large_embedder.tflite` | 10.4MB | FP32 | GPU (MediaPipe) | Generic visual embedding (1280-dim) |
| MobileViTv2-0.75 (embedder) | `mobilevitv2_075_embedder.tflite` | 9.8MB | FP32 | GPU (TFLite) | Alternative generic embedder (canonical-crop work, #91) |
| MobileViTv2-0.75 (classifier) | `mobilevitv2_075_classifier.tflite` | 11.3MB | FP32 | GPU (TFLite) | Online classifier for tentative confirmation |
| VitTracker | `object_tracking_vittrack_2023sep.onnx` | 0.7MB | FP32 | CPU (OpenCV DNN) | Visual frame-to-frame tracker |
| magic_touch | `magic_touch.tflite` | 5.9MB | FP32 | GPU (MediaPipe) | Segmentation for masked crops |
| BlazeFace | `blaze_face_short_range.tflite` | 0.2MB | FP32 | GPU (MediaPipe) | Face detection within person crops |
| age-gender-retail-0013 | `age_gender_retail_0013.tflite` | 4.1MB | FP32 | GPU (fallback CPU) | Face-based gender + age |
| OSNet x1.0 MSMT17 | `osnet_x1_0_msmt17.tflite` | 4.2MB | FP32 | GPU (fallback CPU) | Person re-ID embedding (512-dim) — MSMT17 weights, swapped from Market in #113 |
| MobileFaceNet | `mobilefacenet.tflite` | 5.0MB | FP32 | GPU (fallback CPU) | Face embedding (192-dim) |
| **Total** | | **~63MB** | | |

## Tunable Parameters

All in constructor defaults — no settings UI yet:

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `maxFramesLost` | ReacquisitionEngine | 450 | Frames before giving up (~15s at 30fps) |
| `positionDecayFrames` | ReacquisitionEngine | 30 | Frames for position weight to reach zero |
| `sizeRatioThreshold` | ReacquisitionEngine | 2.0 | Initial max size difference for candidates |
| `minScoreThreshold` | ReacquisitionEngine | 0.45 | Minimum score to accept a candidate |
| `APPEARANCE_OVERRIDE_THRESHOLD` | ReacquisitionEngine | 0.7 | Raw cosine to bypass geometric filters + smart hop threshold |
| `GEOMETRIC_OVERRIDE_THRESHOLD` | ReacquisitionEngine | 0.55 | Lower raw-cosine geometric override (with corroborating signals) |
| `TENTATIVE_BYPASS_THRESHOLD` | ReacquisitionEngine | 0.65 | Raw cosine to bypass tentative-confirmation N-frame wait |
| `Z_GEOMETRIC_OVERRIDE_THRESHOLD` | ReacquisitionEngine | 1.0 | Z-score geometric override (alternative to raw 0.7) |
| `Z_TENTATIVE_BYPASS_THRESHOLD` | ReacquisitionEngine | 1.5 | Z-score tentative bypass |
| `Z_SCORE_MIDPOINT` | ReacquisitionEngine | 1.0 | Sigmoid midpoint for z→[0,1] mapping (#119) |
| `Z_SCORE_SCALE` | ReacquisitionEngine | 1.0 | Sigmoid sharpness for z→[0,1] mapping |
| `Z_SIGMA_FLOOR` | ReacquisitionEngine | 0.05 | Min stddev to avoid divide-by-zero in z-norm |
| `MIN_COHORT_FOR_ZNORM` | ReacquisitionEngine | 5 | Negative samples per modality before z-norm activates |
| `PERSON_REID_FLOOR` | ReacquisitionEngine | 0.45 | Raw OSNet floor on person path |
| `FACE_FLOOR` | ReacquisitionEngine | 0.40 | Raw face floor on person path |
| `ROSTER_REJECT_FLOOR` | ReacquisitionEngine | 0.55 | Min non-lock slot score for roster reject (#113) |
| `ROSTER_REJECT_MARGIN` | ReacquisitionEngine | 0.07 | Margin non-lock slot must beat lock by |
| `SCENE_FACE_MATCH_FLOOR` | ReacquisitionEngine | 0.40 | Scene-face memory match floor (#89) |
| `SCENE_FACE_VETO_MARGIN` | ReacquisitionEngine | 0.05 | Face-only asymmetric veto margin |
| `SCENE_BODY_VETO_MARGIN` | ReacquisitionEngine | 0.10 | Body-only asymmetric veto margin |
| `LOCK_SELF_FLOOR_DEFAULT` | ReacquisitionEngine | 0.5 | Fallback when gallery has < 2 entries (#120) |
| `LOCK_SELF_FLOOR_SCALE` | ReacquisitionEngine | 0.7 | Scale factor on min pairwise self-recall |
| `LOCK_SELF_FLOOR_MIN` | ReacquisitionEngine | 0.25 | Lower clamp on adaptive floor |
| `LOCK_SELF_FLOOR_MAX` | ReacquisitionEngine | 0.40 | Upper clamp on adaptive floor |
| `TEMPLATE_SIM_LOW_RATIO` | ReacquisitionEngine | 0.75 | `templateLow = lockSelfFloor × 0.75` |
| `MAX_GALLERY_SIZE` | ReacquisitionEngine | 12 | Maximum embeddings in the reference gallery |
| `MAX_NEGATIVE_EXAMPLES` | ReacquisitionEngine | 10 | Cap on stored negative impostor embeddings per modality |
| `MAX_SCENE_FACE_PAIRS` | ReacquisitionEngine | 16 | Cap on per-session face/body pair memory |
| `RATIO_TEST_THRESHOLD` | ReacquisitionEngine | 0.85 | Lowe's ratio test threshold (top-1 vs top-2 similarity) |
| `TENTATIVE_MIN_FRAMES` | ReacquisitionEngine | 3 | Frames a candidate must persist before being accepted as a re-lock |
| `TENTATIVE_IOU_THRESHOLD` | ReacquisitionEngine | 0.3 | IoU between tentative frames to count as same candidate |
| `minConfidence` | DetectionFilter | 0.5 | Minimum ML confidence to show detection |
| `maxBoxArea` | DetectionFilter | 0.85 | Reject full-frame detections only; tentative confirmation catches flickering phantoms |
| `minIou` | FrameToFrameTracker | 0.2 | Minimum IoU to match across frames |
| `MIN_CONFIDENCE` | VisualTracker | 0.5 | VitTracker confidence floor |
| `VT_MAX_UNCONFIRMED` | ObjectTracker | 10 | Frames without detector confirmation → drift (secondary signal) |
| `TEMPLATE_CHECK_INTERVAL` | ObjectTracker | 5 | Embed VT crop every N frames for self-verification |
| `TEMPLATE_MISMATCH_MAX` | ObjectTracker | 3 | Consecutive low-sim checks → drift (primary signal) |
| `templateOk` / `templateLow` | ObjectTracker | dynamic | Per-lock from `lockSelfFloor` (no longer static — #120) |
| `VT_BOX_AREA_MAX_RATIO` | ObjectTracker | 5 | VT box area runaway → kill |
| `VT_RESEAT_GROWTH_RATIO` | ObjectTracker | 1.3 | Min growth-since-init for VT reseat (#120) |
| `VT_RESEAT_DISAGREEMENT_RATIO` | ObjectTracker | 1.3 | Min vt-area / det-area for reseat |
| `VT_SKIP_INTERVAL_BASE` | ObjectTracker | 2 | Default skip interval (50% detector frames) |
| `VT_SKIP_INTERVAL_STABLE` | ObjectTracker | 3 | Widened interval when VT is long-stable (33% detector frames) |
| `VT_SKIP_MIN_CONFIRMED` | ObjectTracker | 5 | Min VT confirmations before frame skipping kicks in |
| `VT_SKIP_STABLE_CONFIRMED` | ObjectTracker | 10 | Confirmations required to switch to the stable skip interval |
| `VT_SKIP_STABLE_CONFIDENCE` | ObjectTracker | 0.7 | VT confidence required to switch to the stable skip interval |
| `targetFrameOccupancy` | ZoomController | 0.15 | Target subject size as fraction of frame |
| `zoomSpeed` | ZoomController | 0.05 | Zoom change per frame |
| `scoreThreshold` | ObjectTracker (MediaPipe) | 0.5 | MediaPipe detector confidence cutoff |

## Known sharp edges

- InteractiveSegmenter GPU delegate returns empty masks for crops >100K pixels on Adreno 740. Workaround: downscale to ~300x300 (`MAX_SEGMENT_PIXELS = 90000`).
- TFLite GPU delegate needs both `tensorflow-lite-gpu` and `tensorflow-lite-gpu-api` (`GpuDelegateFactory$Options` lives in the api artifact).
- OSNet on Adreno 740 has overlapping same-person / different-person distributions (`same_p10 ≈ diff_p99`). No fixed cosine threshold separates them — SessionRoster + Phase 4 z-norm are the structural answer.
- OSNet-IBN MSMT17 → TFLite produces degenerate output (#117): InstanceNorm GPU miscompile upstream. OSNet x1.0 MSMT17 (non-IBN) ships instead.
- VitTracker (regression head) lets boxes grow over time even when tracking is correct. Mitigated by detector-anchor reseat + `VT_BOX_AREA_MAX_RATIO` kill, not eliminated.
- VitTracker dies fast on small/transparent objects (bottles, glasses) — confidence drops <0.25 within 2-3s.
- Walking/handheld with mid-recording rotation → upside-down REACQUIRE frames (#116). The runDetector upright transform isn't shared with the rest of the pipeline.
- EfficientDet-Lite2 labels flicker (bowl↔potted plant↔toilet). Handled by the hard person/not-person gate + within-category embedding override; specific labels don't drive matching.
- OpenCV adds ~10MB to APK (arm64).

Roadmap items not yet built: pre-roll buffer, settings UI, landscape UI, battery/thermal management, VT refactor (drift-by-detection / OSTrack / JDE-style joint detector+embedder), photo capture (#38), auto-lock (#35).

## Device Testing

Tested on: Xiaomi Poco F4, Android 15

Wireless debugging disconnects frequently on Xiaomi — this is Xiaomi battery management killing the connection. Workaround: disable battery optimization for wireless debugging, or use USB.
