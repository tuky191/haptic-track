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
├── MainActivity.kt / HapticTrackApp.kt      # Single-activity Compose entry point
├── camera/
│   ├── CameraManager.kt                     # CameraX lifecycle, zoom, recording
│   └── DeviceOrientationListener.kt         # Accelerometer-based rotation detection
├── tracking/                                 # ← most of the complexity lives here
│   ├── ObjectTracker.kt                     # Orchestrator: wires detector, VT, embedder, reacq
│   ├── ReacquisitionEngine.kt               # Cascade scoring: hard category gate, z-norm calibration, ROSTER_REJECT
│   ├── AppearanceEmbedder.kt                # MobileNetV3 embedding + gallery augmentation (generic ID)
│   ├── FaceEmbedder.kt                      # MobileFaceNet (face identity, person path)
│   ├── PersonReIdEmbedder.kt                # OSNet x1.0 MSMT17 (body re-ID, person path)
│   ├── PersonAttributeClassifier.kt         # Crossroad-0230 + BlazeFace + age-gender
│   ├── SessionRoster.kt                     # Per-session non-lock person memory (#108) for ROSTER_REJECT
│   ├── VisualTracker.kt                     # OpenCV VitTracker wrapper
│   ├── FrameToFrameTracker.kt               # IoU-based detection ID assignment
│   ├── VelocityEstimator.kt                 # Per-track velocity from box centers
│   ├── CanonicalCrop.kt                     # Aspect-preserving crop helper for embedders (#100)
│   ├── DetectionFilter.kt                   # Noise removal (confidence, size, aspect ratio; applied before scoring)
│   ├── ScenarioRecorder.kt                  # JSON capture for replay testing + serialization helpers
│   ├── DebugFrameCapture.kt                 # Annotated PNGs + session.log on tracking events
│   ├── CropDebugCapture.kt                  # Per-embedder canonical-crop dump for audit (#92)
│   ├── EmbeddingStabilityLogger.kt          # Diagnostic: tracks per-track sim drift over time
│   ├── FrozenNegatives.kt                   # Cold-start negative pool for the online classifier
│   ├── CropClassifier.kt                    # MobileViTv2 light classifier (online tentative-confirmation)
│   ├── ObjectSegmenter.kt                   # magic_touch segmentation for masked crops
│   ├── EmbeddingUtils.kt                    # Shared: IoU, cosine sim, histogram, model loading, z-norm helpers
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
│   ├── AppearanceEmbedder     (MobileNetV3: generic visual identity fingerprint)
│   ├── FaceEmbedder           (MobileFaceNet: face identity for person path)
│   ├── PersonReIdEmbedder     (OSNet x1.0 MSMT17: body re-ID for person path)
│   ├── PersonAttributeClassifier (Crossroad-0230 + BlazeFace + age-gender: person attributes)
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

## Current Work (may be stale — verify with git/GitHub)

**Recently merged to master (newest first):**
- PR #120 — Partial fix for chair tracking regression (#110): adaptive `lockSelfFloor` + dynamic template thresholds + detector-anchor VT reseat
- PR #119 — Phase 4 (#102): cascade scoring-level z-norm + hard person/not-person category gate
- PR #118 — OSNet GPU miscompile + cache pollution fixes (#116, #117)
- PR #115 — `man_multi_person` scenario tests (#114 phases A + C)
- PR #113 — SessionRoster + OSNet-IBN MSMT17 (Phase 4 of #102, closes #108)
- PR #112 — Re-id score normalization (Z-norm phases 1+3) (#102)
- PR #111 — Tap-to-track auto-starts recording
- PR #105 — Re-id noise floor diagnostics: session.log capture + OSNet gray-bias harness (#102)
- PR #104 — Gallery accumulator: gate on lockSim, drop the size<8 escape hatch (#103)
- PR #101 — Canonical-crop preparation: aspect-preserving inputs across all embedders (#100)
- PR #99 — Audit validation: same-vs-different + crop review (#92 follow-up)
- PR #97 — Embedding-input audit: instrumentation + visible crops (#92)
- PR #89 — Scene face-body memory for asymmetric identity gating (#83 phase 2)
- PR #88 — Ground-truth identity validation for VideoReplayTest
- PR #87 — Synthetic stock-video tests for person identity (#62)
- PR #86 — On-device replay tests via production GL pipeline (#85)
- PR #84 — Identity: face embedding as a gate (#83 phase 1)
- PR #82 — Drift detection: faster template path via decay + lock-only gallery (#80)
- PR #79 — Identity: per-lock adaptive embedding floor + classifier cold-start (#68)
- PR #72 — Identity: gate persons on OSNet, fix self-poisoning negatives (#67)
- PR #61 / #60 — Pipeline perf: bitmap pool, retention, async PNG, PBO readback (#49)
- PR #57 / #55 — Drift detection rewrite (#45): template self-verification primary
- PR #54 — Unified SurfaceTexture pipeline (#48)

**Open issues:**
- **#117** — OSNet-IBN MSMT17 TFLite conversion produces degenerate output (TFLite GPU InstanceNorm miscompile upstream)
- **#116** — Walking session: 12 reacqs in 14s, REACQUIRE frames saved upside-down (rotation pipeline half)
- **#114** — Add real-world multi-person scenario test (post-#113 device session)
- **#110** — chair_living_room tracking rate regressed 76% → 25% (partially closed by #120, residual on threshold floor)
- **#109** — Evaluate EdgeFace-XS as MobileFaceNet replacement
- **#107** — person_playground LOCK records gallery=0 — embedWithAugmentations returning empty
- **#106** — Evaluate OSNet replacement (SOLIDER / MobileCLIP / OSNet-AIN) — gated on #102 residual
- **#98** — Min-bbox-size guard at lock time (#92 follow-up)
- **#94** — Phase 1B (#91): thread masked crop through OSNet (single segmentation pass)
- **#93** — Phase 1A (#91): face alignment via BlazeFace keypoints
- **#90** — Detector label-flicker (person→bed) drops lock on unusual postures
- **#76** — VT init on processing thread (~30-50ms on lock apply)
- **#75** — Camera-switch leaks ~6MB transient viewfinder bitmaps
- **#69** — Identity PR3: replace MobileNetV3-Large with MobileCLIP-S2 / DINOv2-small
- **#65** — FTFTracker ID instability during person search
- **#59** — Small-object rotation tracking (mouse_desk_rotation 31-36%)
- **#51** — Better embedding model + supplementary identity signals
- **#38** — Photo capture + recording speed control
- **#35** — Auto-lock on predefined criteria
- **#27** — Clothing color detection accuracy
- **#21** — Image stabilization
- **#20** — Tracking when phone held upside down (180° rotation)

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
Replaced the 6-signal weighted average with DeepSORT-style cascade gates. Wrong-label candidates are hard-rejected at the label gate (no amount of color/position can rescue them). Within a category, a strong embedding (>0.7) overrides the label gate to handle genuine label flicker. Survivors are ranked with embedding as the primary signal (50%+), not a balanced weight across all signals. This eliminated the chair→person cross-category leakage that plagued the weighted average.

### Hard person/not-person category gate (decided 2026-04-29, PR #119)
Replaced the soft `labelOverride` with a hard person/not-person binary gate. No embedding similarity, no z-score, no position match can rescue a cross-category candidate. The gate fires before any scoring. This was the structural fix for the kid-to-wife / boy-to-chair cases that Phase 1+3 (PR #112) and SessionRoster (PR #113) couldn't close on their own.

### Phase 4 z-norm cascade scoring (decided 2026-04-29, PR #119)
Re-id scores are calibrated per-modality against the live impostor distribution before the cascade ranks them. Raw cosine → z-score → sigmoid. `Z_SCORE_MIDPOINT=1.0`, `Z_SCORE_SCALE=1.0`. A z-score of 1.0 maps to 0.50; z=2.0 → 0.73; z=3.0 → 0.88. Active in `hasFace` and `hasReId` paths once the impostor cohort matures (`MIN_COHORT_FOR_ZNORM=5`). `hasAppearance` (non-person) stays on raw cosine — calibration is person-vs-person. Single-person scenes never mature the cohort, so they fall back to raw cosine.

### SessionRoster: per-session non-lock person memory (decided 2026-04-28, PR #113)
For multi-person scenes, the engine maintains memory of non-lock persons (their face + body embeddings, slot IDs). When a candidate's combined non-lock-slot score beats the lock by `ROSTER_REJECT_MARGIN` (0.07) and exceeds `ROSTER_REJECT_FLOOR` (0.55), the candidate is rejected as a known impostor. Closed #108 — the structural fix needed when no fixed cosine threshold separates same-person from different-person on Adreno 740 (OSNet `same_p10 ≈ diff_p99`).

### Scene face-body memory (decided 2026-04-26, PR #89)
`SessionRoster` is paired memory: each non-lock slot holds both face and body embeddings. When one modality is missing (e.g. detected face but no body crop, or vice versa), the asymmetric veto fires from whichever modality is present, instead of the candidate sneaking through on the missing-modality side.

### Adaptive `lockSelfFloor` (decided 2026-04-30, PR #120)
Replaced the fixed `GALLERY_ADD_LOCK_SIM_FLOOR=0.5` (introduced in PR #103) with a per-lock adaptive floor: `MIN(pairwise self-recall in lock gallery) × 0.7`, clamped to `[0.25, 0.40]`. The fixed 0.5 sat above the chair's 0.30-0.50 cosine band, so the accumulator never grew the gallery past the initial 5 augmentations. Adaptive floor lets weak-embedder classes (chair, mouse) accumulate while still rejecting clearly-different impostors. The same `lockSelfFloor` drives the dynamic `TEMPLATE_SIM_OK` / `TEMPLATE_SIM_LOW` (`× 0.75`) thresholds — drift detection becomes per-lock instead of one-size-fits-all.

### Detector-anchor VT reseat (decided 2026-04-30, PR #120)
VitTracker's regression head can let the box grow over time even when tracking is otherwise correct. When the VT box has both grown ≥1.3× since lock **and** is ≥1.3× the matched detector box, VT is reinitialized on the smallest matched same-category detection instead of being killed. Two conditions are required so that legitimate zoom (subject approaches camera, box grows but detector agrees) doesn't trigger reseat.

### Tap-to-track auto-recording (decided 2026-04-29, PR #111)
The first lock in a session auto-starts video recording so users don't miss the moment between "I see it" and "now I'm pressing record". Volume-down still cycles idle → lock → start recording → stop+clear; tap just starts at step 2 directly.

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

### Volume-down hands-free cycle (decided 2026-04-17, simplified 2026-04-29)
Three-stage cycle on volume-down: idle → lock on center object, tracking → start recording, recording → stop recording + clear. Enables fully hands-free operation: point camera, press volume-down three times. Tap-to-lock now also auto-starts recording on the first lock (PR #111), so the volume-down "lock then start recording" two-press becomes a single press for tap users.

### GPU delegate for all models (decided 2026-04-18, updated 2026-04-19)
All 9 ML models run on GPU. EfficientDet-Lite2 switched from INT8 (CPU) to FP16 (GPU) — downloaded from MediaPipe model zoo, same structure (448x448, 90 classes), fewer spurious detections. TFLite models use `createGpuInterpreter()` which returns `GpuInterpreter` (interpreter + delegate pair) with `Throwable` catch for CPU fallback. MediaPipe models use `Delegate.GPU` via `BaseOptions.setDelegate()`. InteractiveSegmenter GPU has a crop size limit — Adreno 740 returns empty masks above ~100K pixels, so large crops are downscaled to ~300x300 before segmenting (MAX_SEGMENT_PIXELS = 90000).

### Adaptive frame skipping during visual tracking (decided 2026-04-18, widened 2026-04-24)
Two-tier skip interval. **Base regime** (confidence >0.6, confirmed ≥5, 0 unconfirmed): run detector every 2nd frame (~50% skipped). **Stable regime** (confidence >0.7, confirmed ≥10): run detector every 3rd frame (~67% skipped). VT alone runs at ~5ms vs ~35ms for detector. Template self-verification (#45, every 5 frames) is the primary drift signal now, so the detector cadence can be a slower safety net once VT has been stable for a while. Resets on drift, lock clear, or VT stop.

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

301 unit tests, all run via Robolectric (no device needed):

| Class | Tests | What it covers |
|---|---|---|
| `ReacquisitionEngineTest` | 101 | Lock/clear, direct match, cascade gate tests (hard category gate, label gate reject/pass/override, wrong label even with perfect color, embedding ranking, no-embedding fallback, COCO label gate), Phase 4 z-norm scoring, position decay, size threshold decay, timeout, frame counters, appearance embedding (store/clear, similarity scoring, same-label discrimination, fallback without embeddings, weight after decay), appearance override of geometric filters (size, position, weak embedding), two-truck discrimination, visual tracker handoff, label flicker, color histogram scoring, person attribute scoring, face/re-ID tier tests, adaptive `lockSelfFloor` clamping/defaults/reset |
| `ScenarioReplayTest` | 31 | Replay harness validation, real captured scenarios (cup, mouse reacquisition, wrong-category rejection, kid-to-wife panning, man_multi_person, supermarket_checkout), regression baselines with quantitative thresholds |
| `SessionRosterTest` | 22 | Roster slot creation, paired face+body memory, ROSTER_REJECT margin gate, slot pruning, asymmetric veto |
| `ZNormTest` | 16 | Z-norm sigmoid mapping, cohort maturation, calibrated-vs-raw selection, hasReId scoring path uses calibrated z-score |
| `CanonicalCropperTest` | 17 | Aspect-preserving canonical-crop helper for embedders (#100) |
| `RotationRemapTest` | 18 | Coordinate remapping for all orientations (0/90/180/270), center invariance, round-trip correctness |
| `DetectionFilterTest` | 16 | Confidence cutoff, label requirement, box area limits, aspect ratio limits, negative IDs, edge cases |
| `VelocityEstimatorTest` | 14 | Per-track velocity from box centers, decay, reset |
| `OrientationHysteresisTest` | 13 | Hysteresis dead zones, cardinal stability, rapid oscillation, deliberate transitions |
| `FrameToFrameTrackerTest` | 10 | IoU computation, ID assignment, persistence across frames, new object detection, disappearance, reset |
| `PersonAttributesTest` | 19 | Attribute similarity scoring, color matching, raw probability comparison |
| `ZoomControllerTest` | 24 | Zoom in/out/steady, min/max limits, gradual steps, reset, edge proximity, manual zoom (set/clamp/pause/reset) |

On-device instrumentation tests live in `app/src/androidTest/java/`:
- `VideoReplayTest` — video-fed end-to-end runs against captured fixtures, tracking-rate thresholds per scenario
- `OsnetGrayBiasTest` — OSNet output sanity (was the diagnostic harness for #102 noise floor)
- `VideoGLDecoder` — shared helper for piping mp4 through the production GL pipeline

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
| `loadTfliteModel(context, asset)` | `PersonAttributeClassifier`, `FaceEmbedder`, `PersonReIdEmbedder`, `CropClassifier` |
| `cosineSimilarity(a, b)` | `ReacquisitionEngine`, `AppearanceEmbedder` |
| `bestGallerySimilarity(candidate, gallery)` | `ReacquisitionEngine` |
| `minPairwiseSimilarity(gallery)` | `ReacquisitionEngine.computeLockSelfFloor()` (#120) |
| `computeCentroid(gallery)` | `ReacquisitionEngine` (gallery accumulator gate) |
| `l2Normalize(arr)` | Embedders (post-inference normalization) |
| `computeColorHistogram(bitmap, box)` | `ObjectTracker` |
| `histogramCorrelation(a, b)` | `ReacquisitionEngine` |
| `floatArrayToBase64()` / `base64ToFloatArray()` | `ScenarioRecorder`, `ScenarioReplayTest` |

## Models

| Model | File | Size | Precision | Delegate | Purpose |
|---|---|---|---|---|---|
| EfficientDet-Lite2 | `efficientdet-lite2-fp16.tflite` | 11.6MB | FP16 | GPU (MediaPipe) | Primary detector (80 COCO classes, every frame) |
| MobileNetV3 Large | `mobilenet_v3_large_embedder.tflite` | 10.4MB | FP32 | GPU (MediaPipe) | Generic visual embedding (1280-dim) |
| MobileViTv2-0.75 (embedder) | `mobilevitv2_075_embedder.tflite` | 9.8MB | FP32 | GPU (TFLite) | Alternative generic embedder (canonical-crop work, #91) |
| MobileViTv2-0.75 (classifier) | `mobilevitv2_075_classifier.tflite` | 11.3MB | FP32 | GPU (TFLite) | Online classifier for tentative confirmation |
| VitTracker | `object_tracking_vittrack_2023sep.onnx` | 0.7MB | FP32 | CPU (OpenCV DNN) | Visual frame-to-frame tracker |
| magic_touch | `magic_touch.tflite` | 5.9MB | FP32 | GPU (MediaPipe) | Segmentation for masked crops |
| Crossroad-0230 | `person_attributes_crossroad_0230.tflite` | 2.8MB | FP32 | GPU (fallback CPU) | Person body attributes (8 binary) |
| BlazeFace | `blaze_face_short_range.tflite` | 0.2MB | FP32 | GPU (MediaPipe) | Face detection within person crops |
| age-gender-retail-0013 | `age_gender_retail_0013.tflite` | 4.1MB | FP32 | GPU (fallback CPU) | Face-based gender + age |
| OSNet x1.0 MSMT17 | `osnet_x1_0_msmt17.tflite` | 4.2MB | FP32 | GPU (fallback CPU) | Person re-ID embedding (512-dim) — MSMT17 weights, swapped from Market in #113 |
| MobileFaceNet | `mobilefacenet.tflite` | 5.0MB | FP32 | GPU (fallback CPU) | Face embedding (192-dim) |
| **Total** | | **~66MB** | | |

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
- [x] Cascade scoring (hard person/not-person category gate + within-category label gate + Phase 4 z-norm calibration)
- [x] SessionRoster: per-session non-lock person memory + ROSTER_REJECT margin gate (#113)
- [x] Scene face-body paired memory for asymmetric identity veto (#89)
- [x] Adaptive `lockSelfFloor` for accumulator + dynamic template thresholds (#120)
- [x] Detector-anchor VT reseat (handles VitTracker box runaway during legitimate zoom)
- [x] Loading spinner with per-model status during GPU init
- [x] Regression baseline scenarios (31 replay tests with quantitative thresholds)
- [x] On-device video replay (`VideoReplayTest`) feeding production GL pipeline
- [x] 301 unit tests

### Not built yet (from concept doc)
- [ ] **Pre-roll buffer** — continuously capture last 30-60s so you never miss the moment before recording started
- [ ] **Settings UI** — all tunable parameters are constructor defaults with no runtime configuration
- [ ] **Landscape UI** — the activity is portrait-locked. Detection works in all orientations but the UI doesn't rotate
- [ ] **Battery/thermal management** — continuous ML + camera + haptics drains battery fast
- [ ] **VT refactor / drift-by-detection (#120 follow-up)** — VitTracker regression head lets boxes grow over time; evaluate OSTrack/MixFormerV2/SeqTrack or a JDE-style joint detector+embedder
- [ ] **Tracking responsiveness for fast subjects** — kids, pets cause frequent lost/reacquire cycles
- [ ] **Photo capture + recording speed (#38)** — photo interval mode and slow-motion/timelapse
- [ ] **Auto-lock on predefined criteria (#35)** — lock on label + attributes (e.g. "men in 40s") without tap

### Known issues
- InteractiveSegmenter GPU delegate returns empty masks for crops >100K pixels on Adreno 740. Workaround: downscale to ~300x300 before segmenting (MAX_SEGMENT_PIXELS = 90000).
- TFLite GPU delegate requires both `tensorflow-lite-gpu` and `tensorflow-lite-gpu-api` dependencies (GpuDelegateFactory$Options is in the api artifact).
- OSNet on Adreno 740 has overlapping same-person / different-person distributions (`same_p10 ≈ diff_p99`) — no fixed cosine threshold separates them. SessionRoster + Phase 4 z-norm are the structural answer; raw cosine alone is unreliable.
- OSNet-IBN MSMT17 conversion to TFLite produces degenerate output (#117). InstanceNorm GPU miscompile upstream; OSNet x1.0 MSMT17 (non-IBN) ships instead. Tools/conversion preserved in `tools/models/tflite/`.
- VitTracker (regression head) lets boxes grow over time even when tracking is correct. Mitigated by detector-anchor reseat (#120) and `VT_BOX_AREA_MAX_RATIO` kill, but not eliminated — full fix needs a tracker swap.
- VitTracker dies quickly on small/transparent objects (bottles, glasses) — confidence drops below 0.25 within 2-3s, forcing unnecessary re-acquisition cycles.
- Walking/handheld sessions with mid-recording rotation produce upside-down REACQUIRE frames (#116). Detection runs on the rotated bitmap but the rest of the pipeline doesn't share the upright transform.
- Multiple visually similar objects of the same label (several bottles in a bathroom) are hard to distinguish — embedding similarity alone isn't enough.
- EfficientDet-Lite2 labels flicker across frames (bowl↔potted plant↔toilet). The hard person/not-person category gate + within-category label flicker override (raw sim > 0.7) handles this — specific labels don't drive matching anyway.
- Cross-angle re-acquisition weakened when the object looks very different from lock angle. Gallery augmentation + accumulated embeddings help but don't eliminate.
- OpenCV adds ~10MB to APK (arm64).

## Device Testing

Tested on: Xiaomi Poco F4, Android 15

Wireless debugging disconnects frequently on Xiaomi — this is Xiaomi battery management killing the connection. Workaround: disable battery optimization for wireless debugging, or use USB.
