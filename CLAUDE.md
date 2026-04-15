# HapticTrack

Hands-free camera tracking Android app. Uses on-device object detection + haptic feedback + auto-zoom to let you film a subject without looking at the screen.

## Development Workflow

All changes follow this flow — never commit directly to master:

1. **Branch** — create a feature/fix branch from master
2. **PR** — push and open a pull request
3. **Device test** — build, install via ADB, test on physical device
4. **Review** — review the PR (code review or self-review)
5. **Merge** — merge to master and delete the branch

## Quick Reference

```bash
./gradlew assembleDebug          # Build debug APK
./gradlew testDebugUnitTest      # Run unit tests (~3s, 91 tests)
adb install -r app/build/outputs/apk/debug/app-debug.apk  # Deploy to device

adb logcat -d -s "Reacq" -s "VisualTracker" -s "FTFTracker"  # Pull tracking logs
adb logcat -c                                                  # Clear logcat
adb pull /sdcard/Android/data/com.haptictrack/files/debug_frames/  # Pull debug overlays
adb shell rm -f /sdcard/Android/data/com.haptictrack/files/debug_frames/*.png  # Clear debug frames

# Python model quality tests
cd tools && .venv/bin/pytest test_model_quality.py -v
cd tools && .venv/bin/python benchmark_embeddings.py test_fixtures/two_apples/scenario.json /path/to/frames/
```

## Architecture

MVVM with a single-activity Compose UI. Two-tier tracking: visual tracker for frame-to-frame, detector + embeddings for re-acquisition.

```
CameraViewModel
├── CameraManager              (CameraX: preview, image analysis, zoom, video capture)
├── DeviceOrientationListener  (accelerometer: detects upside-down/landscape holds)
├── ObjectTracker              (orchestrates the tracking pipeline below)
│   ├── VisualTracker          (OpenCV TrackerNano: primary frame-to-frame pixel tracking)
│   ├── MediaPipe Detector     (EfficientDet-Lite0: object detection + classification)
│   ├── FrameToFrameTracker    (IoU-based ID assignment for detections)
│   ├── AppearanceEmbedder     (MobileNetV3: visual identity fingerprint)
│   ├── ReacquisitionEngine    (scoring: position + size + label + appearance)
│   ├── DetectionFilter        (noise removal)
│   └── DebugFrameCapture      (saves annotated frames on tracking events)
├── HapticFeedbackManager      (vibration patterns mapped to tracking status)
└── ZoomController             (auto-zoom from bounding box size/position)
```

### Data Flow

**Normal tracking (object visible):**
1. CameraX feeds frames to `ObjectTracker` via `ImageAnalysis`
2. `DeviceOrientationListener` provides physical rotation → bitmap rotated to upright
3. `VisualTracker` (TrackerNano) updates the locked object's position by pixel correlation
4. Detector runs in parallel to cross-check — if detector confirms the label at the tracker's box, tracking continues
5. If detector doesn't confirm for 10 frames → drift detected, visual tracker stopped

**Re-acquisition (object lost):**
1. Detector runs, `FrameToFrameTracker` assigns stable IDs via IoU matching
2. `AppearanceEmbedder` computes visual fingerprints for all candidates
3. `ReacquisitionEngine` scores candidates: position (decays over time) + size + label (20% scoring factor) + appearance similarity (45% weight)
4. Strong embedding match (>0.7 cosine similarity) overrides position/size hard thresholds
5. Best candidate above `minScoreThreshold` becomes the new lock; visual tracker re-initializes

**Display:**
1. `DetectionFilter` removes noise before sending to UI
2. `CameraViewModel` updates `TrackingUiState` (StateFlow), drives haptics + zoom
3. `CameraScreen` renders bounding boxes with FILL_CENTER coordinate transform

## Key Design Decisions

### Two-tier tracking: VisualTracker + Detector (decided 2026-04-13)
TrackerNano (OpenCV) follows the locked object by pixel correlation — no classifier needed, very stable frame-to-frame. But it drifts when the object leaves frame. The detector + embedding pipeline handles re-acquisition. The visual tracker is cross-checked against the detector every frame; 10 consecutive unconfirmed frames triggers drift detection and handoff to re-acquisition.

### Embedding gallery: augmented + accumulated (decided 2026-04-14)
At lock time, `AppearanceEmbedder` generates 5 embeddings (original + rotated 90°/180°/270° + horizontal flip) for immediate multi-angle coverage. During confirmed visual tracking, a new real-world embedding is captured every ~1s. Gallery holds up to 12 embeddings. Re-acquisition compares candidates against the best match in the gallery. This handles viewpoint changes (phone rotation, walking around the object).

### Appearance override of geometric filters (decided 2026-04-13)
When embedding similarity > 0.7, position and size hard thresholds are bypassed. This handles phone rotation (tiny edge-of-frame lock → large centered detection after flip = 12x size ratio) and camera movement (object reappears at completely different screen position).

### Label is a scoring factor, not a gate (decided 2026-04-14)
EfficientDet-Lite0 labels flicker across frames (bowl↔potted plant↔toilet for the same object). A hard label filter caused more harm than good — blocking correct re-acquisition when the detector misclassified. Label is now a 20% weight in scoring. Wrong label loses points but doesn't block. The embedding handles identity; the label is a bonus.

### Confirmed-only position sync (decided 2026-04-13)
`lastKnownBox` only updates from visual tracker when the detector confirms the tracked position. Prevents drifted tracker coordinates from poisoning the re-acquisition search area.

### Device orientation via accelerometer (decided 2026-04-13)
Portrait-locked activity doesn't tell CameraX about upside-down or landscape holds. `DeviceOrientationListener` detects physical rotation (0°/90°/180°/270°), extra rotation applied to bitmap before detection, coordinates remapped back to screen space.

### MediaPipe over ML Kit (decided 2026-04-12)
ML Kit's bounding boxes were too loose — covering the whole desk instead of individual objects. MediaPipe with EfficientDet-Lite0 gives much tighter boxes. Tradeoff: MediaPipe doesn't provide tracking IDs, so we built `FrameToFrameTracker` (IoU-based).

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

## Logging

### Logcat tags

| Tag | What it logs |
|---|---|
| `Reacq` | LOCK, LOST, SEARCH (with candidate scores + similarity), REACQUIRE, TIMEOUT, CLEAR, OVERRIDE (when embedding bypasses geometric filters) |
| `VisualTracker` | INIT (tracker started), LOST (confidence dropped), DRIFT (unconfirmed frames exceeded threshold) |
| `FTFTracker` | NEW (fresh ID assigned), MATCH (IoU match to previous frame) |
| `AppearEmbed` | Embedding failures |
| `DebugCapture` | Saved debug frame filenames |

Filter: `adb logcat -s "Reacq" -s "VisualTracker" -s "FTFTracker"`

### Debug frame capture

Annotated PNGs saved to `/sdcard/Android/data/com.haptictrack/files/debug_frames/` on tracking events:

| Event | When | What's drawn |
|---|---|---|
| LOCK | User taps to lock | Green box on locked object |
| LOST | First frame after losing object | Red dashed box at last known position |
| SEARCH | Same-label candidate exists but not matched, or every 10th search frame | All detections + last known box + candidate info |
| REACQUIRE | Successfully re-locked | Cyan box on re-acquired object |
| TIMEOUT | Gave up searching | Last known box + whatever's in frame |

Auto-prunes to 200 files max.

## Test Suite

91 unit tests, all run via Robolectric (no device needed):

| Class | Tests | What it covers |
|---|---|---|
| `ReacquisitionEngineTest` | 45 | Lock/clear, direct match, re-acquisition scoring, position decay, label as soft scoring factor, size threshold decay, timeout, frame counters, appearance embedding (store/clear, similarity scoring, same-label discrimination, fallback without embeddings, weight after decay), appearance override of geometric filters (size, position, weak embedding), two-truck discrimination, visual tracker handoff, label flicker (strong/weak embedding with mislabeled objects) |
| `DetectionFilterTest` | 15 | Confidence cutoff, label requirement, box area limits, aspect ratio limits, negative IDs, edge cases |
| `FrameToFrameTrackerTest` | 10 | IoU computation, ID assignment, persistence across frames, new object detection, disappearance, reset |
| `RotationRemapTest` | 11 | Coordinate remapping for all orientations (0°, 90°, 180°, 270°), center invariance, round-trip correctness |
| `ZoomControllerTest` | 10 | Zoom in/out/steady, min/max limits, gradual steps, reset, edge proximity |

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

## Models

| Model | File | Size | Purpose |
|---|---|---|---|
| EfficientDet-Lite0 | `efficientdet-lite0.tflite` | 4.4MB | Object detection (80 COCO classes) |
| MobileNetV3 Small | `mobilenet_v3_small_075_224_embedder.tflite` | 3.9MB | Visual embedding (1024-dim, ~4ms/crop) |
| NanoTrack backbone | `nanotrack_backbone.onnx` | 1.0MB | Visual tracker backbone (OpenCV TrackerNano) |
| NanoTrack head | `nanotrack_head.onnx` | 709KB | Visual tracker neck+head |

## Dependencies

| Library | Version | Size impact | Purpose |
|---|---|---|---|
| CameraX | 1.4.1 | — | Camera control + recording |
| MediaPipe Tasks Vision | 0.10.21 | ~15MB | Object detection + image embedding |
| OpenCV | 4.13.0 | ~10MB (arm64) | TrackerNano visual tracking |
| Jetpack Compose BOM | 2024.12.01 | — | UI framework |
| Accompanist Permissions | 0.36.0 | — | Runtime permission handling |

## Tunable Parameters

All in constructor defaults — no settings UI yet:

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `maxFramesLost` | ReacquisitionEngine | 450 | Frames before giving up (~15s at 30fps) |
| `positionDecayFrames` | ReacquisitionEngine | 30 | Frames for position weight to reach zero |
| `sizeRatioThreshold` | ReacquisitionEngine | 2.0 | Initial max size difference for candidates |
| `minScoreThreshold` | ReacquisitionEngine | 0.35 | Minimum score to accept a candidate |
| `APPEARANCE_OVERRIDE_THRESHOLD` | ReacquisitionEngine | 0.7 | Embedding similarity to bypass geometric filters |
| `MAX_GALLERY_SIZE` | ReacquisitionEngine | 12 | Maximum embeddings in the reference gallery |
| `minConfidence` | DetectionFilter | 0.5 | Minimum ML confidence to show detection |
| `minIou` | FrameToFrameTracker | 0.2 | Minimum IoU to match across frames |
| `MIN_CONFIDENCE` | VisualTracker | 0.5 | TrackerNano confidence floor |
| `VT_MAX_UNCONFIRMED` | ObjectTracker | 10 | Frames without detector confirmation → drift |
| `targetFrameOccupancy` | ZoomController | 0.3 | Target subject size as fraction of frame |
| `zoomSpeed` | ZoomController | 0.05 | Zoom change per frame |
| `scoreThreshold` | ObjectTracker (MediaPipe) | 0.5 | MediaPipe detector confidence cutoff |

## What's Built vs. What's Not

### Built (functional)
- [x] Tap-to-lock object selection (smallest-box-wins)
- [x] Two-tier tracking: visual tracker + detector re-acquisition
- [x] Visual embedding for identity-aware re-acquisition
- [x] Haptic feedback (continuous pulse, edge intensity, stop on lost)
- [x] Auto-zoom from bounding box
- [x] Video recording (start/stop)
- [x] All-orientation support (portrait, landscape, upside down)
- [x] Debug frame capture for on-device diagnostics
- [x] Off-device model quality testing (Python + MediaPipe)
- [x] 88 unit tests

### Not built yet (from concept doc)
- [ ] **Pre-roll buffer** — continuously capture last 30-60s so you never miss the moment before recording started. Needs a circular buffer of encoded frames; CameraX doesn't support this natively.
- [ ] **Quick-start gesture** — double-tap volume button to begin tracking + recording instantly. Needs `MediaSession` or accessibility service to intercept volume keys.
- [ ] **Black screen mode** — display off or fake home screen while recording. App store risk (stalkerware policy). Technically: keep camera active via foreground service, suppress UI.
- [ ] **Multi-lens auto-switching** — switch between ultra-wide/wide/telephoto based on zoom level. CameraX supports this on flagship phones via zoom ratio; needs testing per device.
- [ ] **Directional haptic feedback** — vibrate differently based on which direction the subject is drifting (left/right/up/down). Current implementation only has intensity based on edge proximity.
- [ ] **Settings UI** — all tunable parameters are constructor defaults with no runtime configuration.
- [ ] **Custom TFLite models** — fine-tuned detection for specific use cases (pets, sports equipment). Current model uses COCO 80 classes.
- [ ] **Landscape UI** — the activity is portrait-locked. Detection works in all orientations but the UI doesn't rotate.
- [ ] **Battery/thermal management** — continuous ML + camera + haptics drains battery fast. No power optimization yet.
- [ ] **iOS port** — Swift + AVFoundation + Vision framework + Core Haptics. Architecture transfers, APIs don't.

### Known issues
- EfficientDet-Lite0 uses COCO categories (80 classes) — misclassifies unfamiliar objects to nearest match. Labels flicker (bowl↔potted plant↔toilet). Mitigated by soft label scoring + embedding identity.
- Visual tracker drift detection takes ~0.3s (10 frames) — brief green flash on wrong area before correction
- Visually similar objects (bottle/vase) can confuse the embedder at sim ~0.5. Raising override threshold helps but doesn't eliminate.
- Cross-angle re-acquisition weakened when the object looks very different from lock angle (top-down bowl vs side view). Gallery augmentation helps but synthetic rotation ≠ real 3D viewpoint change. Accumulated embeddings during tracking fill this gap over time.
- OpenCV adds ~10MB to APK (arm64). Could be replaced with pure Kotlin correlation tracker to reduce size.

## Device Testing

Tested on: Xiaomi 13 Pro, Android 15 (3.2x optical telephoto)

Wireless debugging disconnects frequently on Xiaomi — this is Xiaomi battery management killing the connection. Workaround: disable battery optimization for wireless debugging, or use USB.
