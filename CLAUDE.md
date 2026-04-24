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
./gradlew testDebugUnitTest      # Run unit tests (~3s, 184 tests)
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
├── MainActivity.kt / HapticTrackApp.kt      # Single-activity Compose entry point
├── camera/
│   ├── CameraManager.kt                     # CameraX lifecycle, zoom, recording
│   └── DeviceOrientationListener.kt         # Accelerometer-based rotation detection
├── tracking/                                 # ← most of the complexity lives here
│   ├── ObjectTracker.kt                     # Orchestrator: wires detector, VT, embedder, reacq
│   ├── ReacquisitionEngine.kt               # Scoring logic (6 weighted signals + overrides)
│   ├── AppearanceEmbedder.kt                # MobileNetV3 embedding + gallery augmentation
│   ├── PersonAttributeClassifier.kt         # Crossroad-0230 + BlazeFace + age-gender
│   ├── VisualTracker.kt                     # OpenCV VitTracker wrapper
│   ├── FrameToFrameTracker.kt               # IoU-based detection ID assignment
│   ├── DetectionFilter.kt                   # Noise removal (confidence, size, aspect ratio; applied before scoring)
│   ├── ScenarioRecorder.kt                  # JSON capture for replay testing + serialization helpers
│   ├── DebugFrameCapture.kt                 # Annotated PNGs + session.log on tracking events
│   ├── ObjectSegmenter.kt                   # magic_touch segmentation for masked crops
│   ├── EmbeddingUtils.kt                    # Shared: IoU, cosine sim, histogram, model loading
│   └── TrackingState.kt                     # Data classes: TrackedObject, PersonAttributes, TrackingUiState
├── ui/
│   ├── CameraScreen.kt                      # Compose UI: viewfinder, bounding boxes, controls
│   └── CameraViewModel.kt                   # MVVM: ties camera, tracker, haptics, zoom
├── haptics/
│   └── HapticFeedbackManager.kt             # Vibration patterns for tracking states
└── zoom/
    └── ZoomController.kt                    # Auto-zoom from bounding box size/position

app/src/test/java/com/haptictrack/tracking/  # Unit tests (Robolectric)
app/src/test/resources/scenarios/            # Captured scenario JSON files for replay
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
│   ├── AppearanceEmbedder     (MobileNetV3: visual identity fingerprint)
│   ├── PersonAttributeClassifier (Crossroad-0230 + BlazeFace + age-gender: person attributes)
│   ├── ReacquisitionEngine    (scoring: position + size + label + appearance + color + attrs)
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
5. Every 5 frames, the current VT crop is embedded and compared against the lock gallery — primary drift signal (catches drift even when the detector can't see the target)
6. 3 consecutive template mismatches (sim < 0.4) OR 10 frames without detector confirmation → drift detected, visual tracker stopped

**Re-acquisition (object lost):**
1. Detector runs, `FrameToFrameTracker` assigns stable IDs via IoU matching
2. `AppearanceEmbedder` computes visual fingerprints for all candidates (async pipeline; synchronous fallback for the closest same-category candidate when the async cache can't bridge across frames during rotation)
3. `DetectionFilter` runs before scoring to strip truly full-frame detections (area > 0.85) — tentative confirmation handles flickering phantoms structurally
4. `ReacquisitionEngine` gates candidates on person/not-person binary category, then ranks on embedding similarity + color histogram + position + size + person attributes
5. Strong embedding match (>0.7 cosine similarity) overrides position/size hard thresholds
6. Best candidate above `minScoreThreshold` becomes the new lock; visual tracker re-initializes
7. `ScenarioRecorder` captures every frame's detections and events as JSON for replay testing

**Recording (4K mode):**
1. Recording toggles VideoCapture on/off — no rebind or mode switching needed
2. Frames always come from the SurfaceTexture GL pipeline, same as normal tracking
3. 4K recording is always available since the pipeline uses only 2 streams (preview + video)

**Stealth mode:**
- Black overlay covers the preview — purely a UI layer (opaque black Box in Compose)
- SurfaceTexture pipeline provides frames regardless of UI visibility
- Volume-down cycle still works: lock → record → stop+clear

**Display:**
1. `DetectionFilter` removes noise before sending to UI
2. `CameraViewModel` updates `TrackingUiState` (StateFlow), drives haptics + zoom
3. `CameraScreen` renders bounding boxes with FILL_CENTER coordinate transform

## Scenario Replay Testing

Every tracking session is automatically recorded as `scenario.json` by `ScenarioRecorder`. This captures the exact inputs to `ReacquisitionEngine.processFrame()` — all detections with their embeddings, color histograms, and person attributes — so any failure can be replayed deterministically as a unit test.

### How it works

1. **On device**: `ScenarioRecorder` activates on lock, records every `processFrame` input, writes `scenario.json` to the session's debug directory alongside PNGs and session.log
2. **Pull**: `adb pull .../debug_frames/session_*/scenario.json`
3. **Add to tests**: copy to `app/src/test/resources/scenarios/`
4. **Write test**: load in `ScenarioReplayTest`, assert expected events

### Workflow: turning a bug into a regression test

```bash
# 1. Reproduce the bug on device
adb install -r app/build/outputs/apk/debug/app-debug.apk
# ... lock on object, trigger the bug, clear

# 2. Pull the scenario
adb pull /sdcard/Android/data/com.haptictrack/files/debug_frames/session_<timestamp>/scenario.json

# 3. Inspect what happened
python3 -c "
import json
d = json.load(open('scenario.json'))
print(f'Lock: {d[\"lock\"][\"label\"]}  Frames: {len(d[\"frames\"])}')
for e in d['events']:
    print(f'  Frame {e[\"frame\"]}: {e[\"type\"]}', e.get('details', ''))
"

# 4. Copy to test resources
cp scenario.json app/src/test/resources/scenarios/descriptive_name.json

# 5. Add a test in ScenarioReplayTest.kt
@Test
fun `descriptive name - asserts expected behavior`() {
    val scenario = loadScenario("descriptive_name.json")
    val result = replay(scenario)
    // Assert: should reacquire the correct object
    assertReacquiresLabel(result, "cup")
    // Assert: should never lock on wrong category
    assertNeverReacquiresLabel(result, "keyboard")
}

# 6. Run tests
./gradlew testDebugUnitTest
```

### Scenario JSON format

```json
{
  "version": 1,
  "lock": {
    "trackingId": 2,
    "label": "cup",
    "cocoLabel": "cup",
    "boundingBox": [0.36, 0.45, 0.63, 0.77],
    "embeddings": ["<base64 IEEE 754 LE>", ...],
    "colorHistogram": "<base64>",
    "personAttributes": null
  },
  "frames": [
    {
      "index": 0,
      "detections": [
        {
          "id": 12,
          "label": "bed",
          "confidence": 0.54,
          "boundingBox": [0.47, 0.38, 0.99, 0.99],
          "embedding": "<base64>",
          "colorHistogram": "<base64>",
          "personAttributes": null
        }
      ]
    }
  ],
  "events": [
    {"frame": 0, "type": "LOST"},
    {"frame": 8, "type": "REACQUIRE", "details": {"id": 18, "label": "cup"}}
  ]
}
```

FloatArrays are encoded as base64 little-endian IEEE 754 (compact: ~1KB for 256-dim embedding). Serialization helpers: `floatArrayToBase64()`, `base64ToFloatArray()`, `jsonToTrackedObject()`, `jsonToPersonAttributes()` in `ScenarioRecorder.kt`.

### Replay test helpers

- `replay(scenario)` — feeds frames through a fresh `ReacquisitionEngine`, returns `ReplayResult` with event list
- `loadScenario(name)` — loads from `src/test/resources/scenarios/`
- `assertReacquiresLabel(result, label)` — assert first reacquisition matches expected label
- `assertNeverReacquiresLabel(result, label)` — assert no reacquisition of a wrong category
- `assertTimesOut(result)` — assert the scenario results in timeout
- `buildSyntheticScenario(...)` — build scenarios programmatically without a device

## Re-acquisition Scoring (cascade gates)

`ReacquisitionEngine.scoreCandidate()` uses DeepSORT-style cascade gates instead of a weighted average. Wrong-category candidates are rejected outright — no amount of color or position similarity can rescue them.

### Gate A: Geometric hard filters

Candidates are rejected if they fail position/size checks (unless embedding override):

- **Position**: center distance > `effectivePositionThreshold` → reject (threshold expands from 0.25 to 1.5 over 30 frames)
- **Size**: size ratio > `effectiveSizeRatioThreshold` → reject (threshold expands over time)
- **Override**: if embedding similarity > 0.7, both position and size filters are bypassed

### Gate B: Label gate

Wrong-label candidates are hard-rejected:

| Condition | Result |
|---|---|
| `lockedLabel == null` | Pass (no label constraint) |
| Label matches `lockedLabel` or `lockedCocoLabel` | Pass |
| Embedding similarity > 0.7 | Pass (label flicker override) |
| Otherwise | **Reject** |

This is the key behavioral difference from the old weighted average: a person with high color similarity to a locked chair is always rejected (unless embedding says it's actually the same object).

### Ranking (among gate survivors)

Survivors are ranked with embedding as the primary signal:

**With embeddings:**
| Signal | Weight | Notes |
|---|---|---|
| Embedding similarity | 50% (+ unused weight) | Primary identity signal |
| Position distance | 15% (decays to 0) | Tiebreaker, meaningless after camera moves |
| Size similarity | 10% | Tiebreaker |
| Color histogram | 15% (when available) | Same-category discrimination |
| Person attributes | 10% (when available) | Same-person discrimination |
| Label bonus | +0.05 | Exact match preferred over embedding-override pass |

**Without embeddings (fallback):**
Position (50%, decays) + size (25%) + base bonus (25%). All survivors already passed the label gate.

### Where the code lives

- `ReacquisitionEngine.scoreCandidate()` — cascade gates + ranking (~line 259)
- `ReacquisitionEngine.findBestCandidate()` — scoring loop, picks highest-ranking survivor (~line 206)
- `ReacquisitionEngine.processFrame()` — top-level: direct match → lost → search → reacquire (~line 124)

## Current Work (may be stale — verify with git/GitHub)

**Recently merged to master:**
- PR #54 — Unify on SurfaceTexture pipeline, remove 3-stream mode (#48)
- PR #52 — Identity: tentative confirmation, ratio test, gallery-relative threshold, online classifier (#50)
- PR #44 — Velocity estimation + async embeddings + SurfaceTexture recording
- PR #42 — GPU delegate for all models, FP16 detector, adaptive frame skip, stealth mode, dependency bumps
- PR #40 — Regression baseline scenarios with quantitative thresholds
- PR #37 — Manual camera controls: pinch zoom, 4K recording, stealth mode, volume-down hands-free cycle
- PR #36 — Person identity: face embedding (MobileFaceNet) + body re-ID (OSNet x1.0)
- PR #31 — Cascade scoring refactor (#30 phases 1+2), scenario recorder + replay harness
- PR #29 — Two-stage detection (EfficientDet-Lite2 + YOLOv8n-oiv7 label enrichment)

**Open issues:**
- **#43** — Tracking responsiveness for fast-moving subjects (kids, pets)
- **#38** — Photo capture + recording speed control
- **#35** — Auto-lock on predefined criteria (label + attributes)
- **#30** — Phase 3: more scenario captures
- **#20** — Upside-down tracking
- **#21** — Image stabilization
- **#27** — Clothing color accuracy

## Key Design Decisions

### Single-stage detection: EfficientDet-Lite2 (decided 2026-04-17, simplified 2026-04-24)
EfficientDet-Lite2 (COCO 80) runs every frame for reliable bounding boxes. YOLOv8n-oiv7 label enrichment was removed once the re-acquisition gate became binary person/not-person — the finer OIV7 label no longer influenced any gate or score, only the UI display text. Removing saved ~270ms at lock time, ~1-2s at app startup, 6.8MB APK size, and one GPU interpreter. `PersonAttributeClassifier` handles person-specific attributes (gender, clothing, age) where OIV7 labels would have been too coarse anyway.

### Two-tier tracking: VisualTracker + Detector (decided 2026-04-13, revised 2026-04-24)
VitTracker (OpenCV) follows the locked object by pixel correlation — no classifier needed, very stable frame-to-frame. But it drifts when the object leaves frame. The detector + embedding pipeline handles re-acquisition.

**Drift detection uses two parallel signals:**
1. **Template self-verification (primary)**: every 5 frames during VT tracking, embed the current VT crop (raw crop fallback, ~8ms) and compare to the lock gallery via `bestGallerySimilarity()`. 3 consecutive similarities below 0.4 (~0.5s) triggers drift. Detector-independent — catches drift even when the detector can't see the target (small objects, uniform surfaces, label flicker).
2. **Detector cross-check (secondary)**: if the detector doesn't confirm the VT position for 10 consecutive frames, also triggers drift. Kept as a safety net for cases where the template somehow keeps matching but the object has moved.

### Template self-verification rationale (decided 2026-04-24)
The old approach (detector cross-check only) had a structural flaw: "detector found something elsewhere in the frame but not at VT position" was counted as drift evidence, even when the detector simply missed our object (common for small/uniform objects like a yellow bowl). This caused false drift kills on correctly-tracking VT. Research (TLD PAMI 2012, Stark ICCV 2021, DaSiamRPN ECCV 2018) consistently treats the tracker's own self-assessment as the primary drift signal, with the detector as a peer rather than a judge. Template self-verification directly measures "does this crop still look like the locked object" — the right question.

### Cascade scoring: label as gate, not weight (decided 2026-04-17)
Replaced the 6-signal weighted average with DeepSORT-style cascade gates. Wrong-label candidates are hard-rejected at the label gate (no amount of color/position can rescue them). Only a strong embedding (>0.7) overrides the label gate, handling genuine label flicker. Survivors are ranked with embedding as the primary signal (50%+), not a balanced weight across all signals. This eliminates the chair→person cross-category leakage that plagued the weighted average.

### Embedding gallery: augmented + accumulated (decided 2026-04-14)
At lock time, `AppearanceEmbedder` generates 5 embeddings (original + rotated 90/180/270 + horizontal flip) for immediate multi-angle coverage. During confirmed visual tracking, a new real-world embedding is captured every ~1s. Gallery holds up to 12 embeddings. Re-acquisition compares candidates against the best match in the gallery.

### Appearance override of geometric filters (decided 2026-04-13)
When embedding similarity > 0.7, position and size hard thresholds are bypassed. This handles phone rotation (tiny edge-of-frame lock → large centered detection after flip = 12x size ratio) and camera movement (object reappears at completely different screen position).

### Label is a gate with embedding override (decided 2026-04-17, revised from 2026-04-14)
Originally label was a soft scoring factor (20% weight) because EfficientDet labels flicker. This led to cross-category leakage. Now label is a hard gate — wrong label is rejected. EfficientDet label flicker is handled by the embedding override: if sim > 0.7, the label gate is bypassed (same object, flickered label).

### Confirmed-only position sync (decided 2026-04-13)
`lastKnownBox` only updates from visual tracker when the detector confirms the tracked position. Prevents drifted tracker coordinates from poisoning the re-acquisition search area.

### Device orientation via accelerometer (decided 2026-04-13)
Portrait-locked activity doesn't tell CameraX about upside-down or landscape holds. `DeviceOrientationListener` detects physical rotation (0/90/180/270), extra rotation applied to bitmap before detection, coordinates remapped back to screen space.

### MediaPipe over ML Kit (decided 2026-04-12)
ML Kit's bounding boxes were too loose — covering the whole desk instead of individual objects. MediaPipe with EfficientDet gives much tighter boxes. Tradeoff: MediaPipe doesn't provide tracking IDs, so we built `FrameToFrameTracker` (IoU-based).

### Immutable `lockedLabel` for re-acquisition (decided 2026-04-12)
When you tap to lock, the label at that moment is saved as `lockedLabel` and never updated. Re-acquisition only considers candidates matching this label. This prevents label drift.

### Position weight decays over time (decided 2026-04-12)
Handheld cameras move. After losing an object, its screen position becomes meaningless within ~1 second. `ReacquisitionEngine` decays position weight to zero over `positionDecayFrames` (30 frames).

### Smallest-box-wins tap selection (decided 2026-04-12)
When multiple bounding boxes overlap at the tap point, the smallest box is selected. This prevents accidentally locking onto background surfaces.

### FILL_CENTER coordinate transform (decided 2026-04-12)
Camera image aspect ratio differs from phone screen. All bounding box rendering and tap handling goes through `FillCenterTransform`.

### Confidence threshold at 50% (decided 2026-04-12)
Both MediaPipe's detector and `DetectionFilter` use 0.5 minimum confidence.

### Unified SurfaceTexture pipeline (decided 2026-04-23, replaces 2/3-stream switching)
Always uses 2-stream mode (SurfaceTexture + VideoCapture). Frames for tracking come from the GL pipeline via SurfaceTexture, not ImageAnalysis. Recording just toggles VideoCapture on/off — no rebind, no mode switching, no `prepareForRebind()`. This replaced the old dynamic 2/3-stream switching where ImageAnalysis was dropped during recording and `PreviewView.getBitmap()` was used as a fallback frame source. The unified pipeline is simpler and always delivers 4K-capable streams.

### Stealth mode: black overlay over SurfaceTexture preview (decided 2026-04-17, updated 2026-04-23)
Opaque black Box overlays the preview in Compose. SurfaceTexture pipeline provides frames regardless of UI visibility, so stealth is purely a UI concern. User sees black screen with controls.

### Volume-down hands-free cycle (decided 2026-04-17)
Three-stage cycle on volume-down: idle → lock on center object, tracking → start recording, recording → stop recording + clear. Enables fully hands-free operation: point camera, press volume-down three times.

### GPU delegate for all models (decided 2026-04-18, updated 2026-04-19)
All 9 ML models run on GPU. EfficientDet-Lite2 switched from INT8 (CPU) to FP16 (GPU) — downloaded from MediaPipe model zoo, same structure (448x448, 90 classes), fewer spurious detections. TFLite models use `createGpuInterpreter()` which returns `GpuInterpreter` (interpreter + delegate pair) with `Throwable` catch for CPU fallback. MediaPipe models use `Delegate.GPU` via `BaseOptions.setDelegate()`. InteractiveSegmenter GPU has a crop size limit — Adreno 740 returns empty masks above ~100K pixels, so large crops are downscaled to ~300x300 before segmenting (MAX_SEGMENT_PIXELS = 90000).

### Adaptive frame skipping during visual tracking (decided 2026-04-18)
When VitTracker is confident (>0.6) and has 5+ confirmed frames with 0 unconfirmed, skip the EfficientDet detector every other frame. VT alone runs at ~5ms vs ~35ms for detector. Still runs detector every 2nd frame for drift detection. Saves ~50% detector compute while locked. Resets on drift, lock clear, or VT stop.

## Logging

### Logcat tags

| Tag | What it logs |
|---|---|
| `Reacq` | LOCK, LOST, SEARCH (with candidate scores + similarity), REACQUIRE (with hop count + sim), TIMEOUT, CLEAR, GIVE_UP, OVERRIDE (when embedding bypasses geometric filters) |
| `VisualTracker` | INIT (tracker started), LOST (confidence dropped), DRIFT (template mismatch OR unconfirmed frames exceeded), Template mismatch (per-check log when sim below threshold) |
| `FTFTracker` | NEW (fresh ID assigned), MATCH (IoU match to previous frame) |
| `ScenarioRec` | Recording started/stopped, frame count |
| `AppearEmbed` | Embedding failures, gallery additions |
| `EmbedSync` | Synchronous embedding fallback fired for a candidate that missed the async cache |
| `TFLiteGPU` | GPU delegate activation or CPU fallback per model |
| `DebugCapture` | Saved debug frame filenames |

Filter: `adb logcat -s "Reacq" -s "VisualTracker" -s "FTFTracker"`

### Debug frame capture

Each tracking session gets its own directory under `/sdcard/Android/data/com.haptictrack/files/debug_frames/`:

```
session_20260417_102137_cup/
├── 102137_151_LOCK.png          # Annotated frame
├── 102137_151_LOCK_raw.png      # Raw frame (no annotations)
├── 102139_812_LOST.png
├── 102141_710_REACQUIRE.png
├── ...
├── session.log                  # All log messages for this session
└── scenario.json                # Full processFrame inputs for replay testing
```

| Event | When | What's drawn |
|---|---|---|
| LOCK | User taps to lock | Green box on locked object |
| LOST | First frame after losing object | Red dashed box at last known position |
| SEARCH | Same-label candidate exists but not matched, or every 10th search frame | All detections + last known box + candidate info |
| REACQUIRE | Successfully re-locked | Cyan box on re-acquired object |
| TIMEOUT | Gave up searching | Last known box + whatever's in frame |

Auto-prunes to 10 sessions max.

## Test Suite

183 unit tests, all run via Robolectric (no device needed):

| Class | Tests | What it covers |
|---|---|---|
| `ReacquisitionEngineTest` | 74 | Lock/clear, direct match, cascade gate tests (label gate reject/pass/override, wrong label even with perfect color, embedding ranking, no-embedding fallback, COCO label gate), position decay, size threshold decay, timeout, frame counters, appearance embedding (store/clear, similarity scoring, same-label discrimination, fallback without embeddings, weight after decay), appearance override of geometric filters (size, position, weak embedding), two-truck discrimination, visual tracker handoff, label flicker (strong/weak embedding with mislabeled objects), color histogram scoring, person attribute scoring, face/re-ID tier tests |
| `ScenarioReplayTest` | 15 | Replay harness validation, real captured scenarios (cup, mouse reacquisition, wrong-category rejection), regression baselines with quantitative thresholds (boy label flicker 67% tracking, person recovery 89% tracking, person↔boy flicker, chair no false-reacquire) |
| `DetectionFilterTest` | 15 | Confidence cutoff, label requirement, box area limits, aspect ratio limits, negative IDs, edge cases |
| `OrientationHysteresisTest` | 13 | Hysteresis dead zones, cardinal stability, rapid oscillation, deliberate transitions |
| `FrameToFrameTrackerTest` | 10 | IoU computation, ID assignment, persistence across frames, new object detection, disappearance, reset |
| `RotationRemapTest` | 11 | Coordinate remapping for all orientations (0, 90, 180, 270), center invariance, round-trip correctness |
| `PersonAttributesTest` | 30 | Attribute similarity scoring, color matching, raw probability comparison |
| `ZoomControllerTest` | 20 | Zoom in/out/steady, min/max limits, gradual steps, reset, edge proximity, manual zoom (set/clamp/pause/reset) |

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

## Shared Utilities (EmbeddingUtils.kt)

Top-level functions used across multiple classes — avoids duplication:

| Function | Used by |
|---|---|
| `computeIou(a, b)` | `FrameToFrameTracker` (delegates), `ObjectTracker` |
| `loadTfliteModel(context, asset)` | `PersonAttributeClassifier` |
| `cosineSimilarity(a, b)` | `ReacquisitionEngine`, `AppearanceEmbedder` |
| `bestGallerySimilarity(candidate, gallery)` | `ReacquisitionEngine` |
| `computeColorHistogram(bitmap, box)` | `ObjectTracker` |
| `histogramCorrelation(a, b)` | `ReacquisitionEngine` |
| `floatArrayToBase64()` / `base64ToFloatArray()` | `ScenarioRecorder`, `ScenarioReplayTest` |

## Models

| Model | File | Size | Precision | Delegate | Purpose |
|---|---|---|---|---|---|
| EfficientDet-Lite2 | `efficientdet-lite2-fp16.tflite` | 11.6MB | FP16 | GPU (MediaPipe) | Primary detector (80 COCO classes, every frame) |
| MobileNetV3 Large | `mobilenet_v3_large_embedder.tflite` | 10MB | FP32 | GPU (MediaPipe) | Visual embedding (1280-dim) |
| VitTracker | `vitTracker.onnx` | 0.7MB | FP32 | CPU (OpenCV DNN) | Visual frame-to-frame tracker |
| magic_touch | `magic_touch.tflite` | 5.9MB | FP32 | GPU (MediaPipe) | Segmentation for masked crops |
| Crossroad-0230 | `person_attributes_crossroad_0230.tflite` | 2.8MB | FP32 | GPU (fallback CPU) | Person body attributes (8 binary) |
| BlazeFace | `blaze_face_short_range.tflite` | 0.2MB | FP32 | GPU (MediaPipe) | Face detection within person crops |
| age-gender-retail-0013 | `age_gender_retail_0013.tflite` | 4.1MB | FP32 | GPU (fallback CPU) | Face-based gender + age |
| OSNet x1.0 | `osnet_x1_0_market.tflite` | 4.2MB | FP32 | GPU (fallback CPU) | Person re-ID embedding (512-dim) |
| MobileFaceNet | `mobilefacenet.tflite` | 5.0MB | FP32 | GPU (fallback CPU) | Face embedding (192-dim) |
| **Total** | | **~40MB** | | |

## Dependencies

| Library | Version | Size impact | Purpose |
|---|---|---|---|
| CameraX | 1.6.0 | — | Camera control + recording |
| MediaPipe Tasks Vision | 0.10.33 | ~15MB | Object detection + image embedding |
| OpenCV | 4.13.0 | ~10MB (arm64) | VitTracker visual tracking |
| TensorFlow Lite | 2.17.0 | — | PersonAttributeClassifier / face / re-ID inference |
| TensorFlow Lite GPU | 2.17.0 | — | GPU delegate for FP32/FP16 TFLite models |
| Jetpack Compose BOM | 2026.03.01 | — | UI framework |
| Accompanist Permissions | 0.37.3 | — | Runtime permission handling |

## Tunable Parameters

All in constructor defaults — no settings UI yet:

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `maxFramesLost` | ReacquisitionEngine | 450 | Frames before giving up (~15s at 30fps) |
| `positionDecayFrames` | ReacquisitionEngine | 30 | Frames for position weight to reach zero |
| `sizeRatioThreshold` | ReacquisitionEngine | 2.0 | Initial max size difference for candidates |
| `minScoreThreshold` | ReacquisitionEngine | 0.45 | Minimum score to accept a candidate |
| `APPEARANCE_OVERRIDE_THRESHOLD` | ReacquisitionEngine | 0.7 | Embedding similarity to bypass geometric filters + smart hop threshold |
| `MAX_GALLERY_SIZE` | ReacquisitionEngine | 12 | Maximum embeddings in the reference gallery |
| `minConfidence` | DetectionFilter | 0.5 | Minimum ML confidence to show detection |
| `maxBoxArea` | DetectionFilter | 0.85 | Reject full-frame detections only; tentative confirmation catches flickering phantoms |
| `minIou` | FrameToFrameTracker | 0.2 | Minimum IoU to match across frames |
| `MIN_CONFIDENCE` | VisualTracker | 0.5 | VitTracker confidence floor |
| `VT_MAX_UNCONFIRMED` | ObjectTracker | 10 | Frames without detector confirmation → drift (secondary signal) |
| `TEMPLATE_CHECK_INTERVAL` | ObjectTracker | 5 | Embed VT crop every N frames for self-verification |
| `TEMPLATE_SIM_THRESHOLD` | ObjectTracker | 0.4 | Similarity below this = drift suspicion |
| `TEMPLATE_MISMATCH_MAX` | ObjectTracker | 3 | Consecutive low-sim checks → drift (primary signal) |
| `targetFrameOccupancy` | ZoomController | 0.15 | Target subject size as fraction of frame |
| `zoomSpeed` | ZoomController | 0.05 | Zoom change per frame |
| `scoreThreshold` | ObjectTracker (MediaPipe) | 0.5 | MediaPipe detector confidence cutoff |
| `VT_SKIP_INTERVAL` | ObjectTracker | 2 | Run detector every Nth frame when VT is skipping |
| `VT_SKIP_MIN_CONFIRMED` | ObjectTracker | 5 | Min VT confirmations before frame skipping kicks in |

## What's Built vs. What's Not

### Built (functional)
- [x] Tap-to-lock object selection (smallest-box-wins)
- [x] Two-tier tracking: visual tracker + detector re-acquisition
- [x] Visual embedding for identity-aware re-acquisition
- [x] Person identity: face embedding + body re-ID for person-vs-person discrimination
- [x] Person attribute classification (gender, clothing, accessories, age)
- [x] Color histogram scoring for same-category discrimination
- [x] Haptic feedback (continuous pulse, edge intensity, stop on lost)
- [x] Auto-zoom from bounding box + pinch-to-zoom with manual override
- [x] Volume-down hands-free cycle: lock center → record → stop+clear
- [x] 4K video recording (unified SurfaceTexture pipeline)
- [x] Stealth mode (true stealth: blank screen, tap to exit, volume-down only)
- [x] All-orientation support (portrait, landscape, upside down) with hysteresis
- [x] GPU delegate for all 9 models (FP16 detector + FP32 others on Adreno 740)
- [x] Adaptive frame skipping (skip detector when VT confident)
- [x] Debug frame capture for on-device diagnostics
- [x] Scenario recording + deterministic replay testing
- [x] Off-device model quality testing (Python + MediaPipe)
- [x] Cascade scoring (DeepSORT-style label gate + embedding-primary ranking)
- [x] Loading spinner with per-model status during GPU init
- [x] Regression baseline scenarios (15 replay tests with quantitative thresholds)
- [x] 184 unit tests

### Not built yet (from concept doc)
- [ ] **Scenario replay validation (#30 phase 3)** — capture more real-world scenarios and validate cascade behavior against them. Phase 1 (recorder) and Phase 2 (cascade) are done.
- [ ] **Pre-roll buffer** — continuously capture last 30-60s so you never miss the moment before recording started
- [ ] **Quick-start gesture** — double-tap volume button to begin tracking + recording instantly
- [ ] **Settings UI** — all tunable parameters are constructor defaults with no runtime configuration
- [ ] **Landscape UI** — the activity is portrait-locked. Detection works in all orientations but the UI doesn't rotate
- [ ] **Battery/thermal management** — continuous ML + camera + haptics drains battery fast
- [ ] **Tracking responsiveness (#43)** — fast-moving subjects (kids, pets) cause frequent lost/reacquire cycles. VT drift detection too aggressive, re-acquisition too slow, auto-zoom can't keep up.
- [ ] **Photo capture + recording speed (#38)** — photo interval mode and slow-motion/timelapse

### Known issues
- InteractiveSegmenter GPU delegate returns empty masks for crops >100K pixels on Adreno 740. Workaround: downscale to ~300x300 before segmenting (MAX_SEGMENT_PIXELS = 90000).
- TFLite GPU delegate requires both `tensorflow-lite-gpu` and `tensorflow-lite-gpu-api` dependencies (GpuDelegateFactory$Options is in the api artifact).
- Wrong-label candidates with strong embedding (>0.7) still pass the label gate — this handles genuine label flicker but could theoretically allow cross-category matches if the embedder is confused. In practice, sim>0.7 across categories is rare.
- Visual tracker (VitTracker) dies quickly on small/transparent objects (bottles, glasses) — confidence drops below 0.25 within 2-3s, forcing unnecessary re-acquisition cycles.
- Multiple visually similar objects of the same label (several bottles in a bathroom) are hard to distinguish — embedding similarity alone isn't enough.
- EfficientDet-Lite2 labels flicker across frames (bowl↔potted plant↔toilet). Mitigated by the binary person/not-person gate + embedding identity — specific labels don't drive matching anyway.
- Cross-angle re-acquisition weakened when the object looks very different from lock angle. Gallery augmentation + accumulated embeddings help but don't eliminate.
- OpenCV adds ~10MB to APK (arm64).

## Device Testing

Tested on: Xiaomi Poco F4, Android 15

Wireless debugging disconnects frequently on Xiaomi — this is Xiaomi battery management killing the connection. Workaround: disable battery optimization for wireless debugging, or use USB.
