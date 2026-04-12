# HapticTrack

Hands-free camera tracking Android app. Uses on-device object detection + haptic feedback + auto-zoom to let you film a subject without looking at the screen.

## Quick Reference

```bash
./gradlew assembleDebug          # Build debug APK
./gradlew testDebugUnitTest      # Run unit tests (~3s)
adb exec-out screencap -p > /tmp/screen.png  # Screenshot from device
adb logcat -d -s "Reacq" -s "FTFTracker"     # Pull tracking decision logs
adb logcat -c                                  # Clear logcat
```

## Architecture

MVVM with a single-activity Compose UI. The ViewModel orchestrates four independent subsystems:

```
CameraViewModel
├── CameraManager        (CameraX: preview, image analysis, zoom, video capture)
├── ObjectTracker        (MediaPipe → FrameToFrameTracker → ReacquisitionEngine → DetectionFilter)
├── HapticFeedbackManager (vibration patterns mapped to tracking status)
└── ZoomController       (auto-zoom from bounding box size/position)
```

### Data Flow

1. CameraX feeds frames to `ObjectTracker` via `ImageAnalysis`
2. MediaPipe runs EfficientDet-Lite0, returns per-frame detections (no tracking IDs)
3. `FrameToFrameTracker` assigns stable IDs via IoU matching across frames
4. `ReacquisitionEngine` handles lock/loss/re-acquisition logic
5. `DetectionFilter` removes noise before sending to UI
6. `CameraViewModel` updates `TrackingUiState` (StateFlow), drives haptics + zoom
7. `CameraScreen` renders bounding boxes with FILL_CENTER coordinate transform

## Key Design Decisions

### MediaPipe over ML Kit (decided 2025-04-12)
ML Kit's bounding boxes were too loose — covering the whole desk instead of individual objects. MediaPipe with EfficientDet-Lite0 gives much tighter boxes. Tradeoff: MediaPipe doesn't provide tracking IDs, so we built `FrameToFrameTracker` (IoU-based).

### Immutable `lockedLabel` for re-acquisition (decided 2025-04-12)
When you tap to lock, the label at that moment is saved as `lockedLabel` and never updated. Re-acquisition only considers candidates matching this label. This prevents label drift — if the engine briefly follows a wrong object, it won't contaminate the search criteria. `lastKnownLabel` still updates for display purposes.

### Position weight decays over time (decided 2025-04-12)
Handheld cameras move. After losing an object, its screen position becomes meaningless within ~1 second. `ReacquisitionEngine` decays position weight to zero over `positionDecayFrames` (30 frames). After decay, only label + size matter. Position threshold also expands from 0.25 to 1.5 (full frame). Size ratio threshold similarly relaxes over time.

### Smallest-box-wins tap selection (decided 2025-04-12)
When multiple bounding boxes overlap at the tap point (e.g., a cup on a table), the smallest box is selected. This prevents accidentally locking onto background surfaces.

### FILL_CENTER coordinate transform (decided 2025-04-12)
The camera image aspect ratio differs from the phone screen. `FILL_CENTER` crops the image to fill the screen. All bounding box rendering and tap handling goes through `FillCenterTransform` to map between normalized image coordinates and screen pixels.

### Confidence threshold at 50% (decided 2025-04-12)
Both MediaPipe's detector and `DetectionFilter` use 0.5 minimum confidence. Lower values produce too many false classifications (pen holders as "cups", cables as "toothbrushes").

## Logging

The `ReacquisitionEngine` logs every decision to Android logcat under tag `Reacq`:
- `LOCK` — user tapped to lock, includes id, label, box, size
- `LOST` — tracking ID disappeared from frame
- `SEARCH` — periodic (every 10 frames) dump of candidates, scores, rejections
- `REACQUIRE` — successfully re-locked onto a new detection
- `TIMEOUT` — gave up after maxFramesLost frames
- `CLEAR` — user pressed Clear

`FrameToFrameTracker` logs under tag `FTFTracker`:
- `NEW` — fresh tracking ID assigned
- `MATCH` — detection matched to previous frame via IoU

Filter logcat: `adb logcat -s "Reacq" -s "FTFTracker"`

## Test Suite

59 unit tests, all run via Robolectric (no device needed):

| Class | Tests | What it covers |
|---|---|---|
| `ReacquisitionEngineTest` | 28 | Lock/clear, direct match, re-acquisition scoring, position decay, label hard filter, size threshold decay, timeout, frame counters |
| `DetectionFilterTest` | 15 | Confidence cutoff, label requirement, box area limits, aspect ratio limits, negative IDs, edge cases |
| `FrameToFrameTrackerTest` | 10 | IoU computation, ID assignment, persistence across frames, new object detection, disappearance, reset |
| `ZoomControllerTest` | 10 | Zoom in/out/steady, min/max limits, gradual steps, reset, edge proximity |

## Current Limitations & Next Steps

### Known issues
- EfficientDet-Lite0 uses COCO categories (80 classes) — misclassifies unfamiliar objects to nearest match (ethernet module → "laptop", spoon → "toothbrush")
- Frequent brief LOST→REACQUIRE cycles when the detector flickers between frames
- No pre-roll buffer yet (concept doc specifies 30-60s)
- No quick-start gesture (volume button double-tap)
- Portrait orientation only

### Tunable parameters
All in constructor defaults — no settings UI yet:

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `maxFramesLost` | ReacquisitionEngine | 90 | Frames before giving up (~3s at 30fps) |
| `positionDecayFrames` | ReacquisitionEngine | 30 | Frames for position weight to reach zero |
| `sizeRatioThreshold` | ReacquisitionEngine | 2.0 | Initial max size difference for candidates |
| `minScoreThreshold` | ReacquisitionEngine | 0.35 | Minimum score to accept a candidate |
| `minConfidence` | DetectionFilter | 0.5 | Minimum ML confidence to show detection |
| `minIou` | FrameToFrameTracker | 0.2 | Minimum IoU to match across frames |
| `targetFrameOccupancy` | ZoomController | 0.3 | Target subject size as fraction of frame |
| `zoomSpeed` | ZoomController | 0.05 | Zoom change per frame |
| `scoreThreshold` | ObjectTracker (MediaPipe) | 0.5 | MediaPipe detector confidence cutoff |

### Model alternatives
- `efficientdet-lite2.tflite` — more accurate but slower, worth testing
- Custom TFLite model fine-tuned for specific use cases (pets, sports equipment)

## Device Testing

Tested on: Xiaomi 2210132G (Poco F4), Android 15

Wireless debugging disconnects frequently on Xiaomi — this is Xiaomi battery management killing the connection. Workaround: disable battery optimization for wireless debugging, or use USB.
