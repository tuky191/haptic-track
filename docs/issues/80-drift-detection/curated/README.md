# Issue #80 — Drift detection delay (slow vs fast contrast)

Two captured sessions from 2026-04-25, same device, same code (`a6283daa` + #79's adaptive-floor branch), back-to-back. The slow case took ~7s to declare drift; the fast case took ~3s. The difference was *what the camera panned to after the subject left view*, not the code.

## `slow_7s/` — hysteresis reset masks template self-verification

Locked person, panned camera away to a uniform indoor backdrop. Template self-verification kept incrementing then resetting; eventually drift was declared via the slower detector cross-check (`VT_MAX_UNCONFIRMED = 10`).

Key timeline (full trace in `slow_7s/session.log`):
```
115018_971  LOCK on person
115022_xxx  Template mismatch 1/3 — counter reset before reaching 3
115023_xxx  Template mismatch 1/3 again — same outcome
115028_091  DRIFT detected, 11 unconfirmed frames (~7s after lock)
```

`lock.png` shows the person at lock time. `lost_at_drift.png` shows the scene when DRIFT finally fired — a uniform indoor backdrop similar enough to gallery views that template sim flickered between 0.3 and 0.5.

## `fast_3s/` — clean 3-in-a-row template trip

Same gestalt — locked a person, panned away — but the camera landed on visually distinct content. Template sim stayed consistently low (0.13, 0.12, 0.20), the counter advanced cleanly, and DRIFT tripped at exactly 3 mismatches.

Key timeline (full trace in `fast_3s/session.log`):
```
115418_791  LOCK on person
115420_892  Template mismatch 1/3 (sim=0.131)
115421_202  Template mismatch 2/3 (sim=0.123)
115421_724  Template mismatch 3/3 (sim=0.198)
115421_724  DRIFT — template mismatch 3x  ← ~3s after lock
```

## What the contrast tells us

The bug isn't "drift detection always slow." It's "drift detection slow when the empty-frame VT crop happens to land on something visually similar to the gallery, even by coincidence." When the VT crop sits on uniform backdrop (white wall, sky, evenly-lit floor), sim can flicker between 0.3 and 0.5 because the crop's color/texture statistics happen to overlap with backgrounds the gallery saw. That's exactly the case where hysteresis-with-reset behaves worst — every fluky high-sim frame undoes the prior mismatch counter.

The structural fix proposed in option (A) of the issue body — decay the counter instead of resetting it — handles both cases:

- **`fast_3s/` (consistent low sim)**: counter advances cleanly, behavior unchanged.
- **`slow_7s/` (intermittent flicker)**: counter survives single high-sim flukes, trips at 3 mismatches over a slightly larger window instead of getting stuck.
