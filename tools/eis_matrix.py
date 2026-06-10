#!/usr/bin/env python3
"""Phase 1 experiment-matrix evaluator.

Roles (same scene, ~20s each):
  A braced,   EIS off  -> noise floor
  B braced,   EIS on   -> injection test       PASS: residual <= 1.1x A per band
  C handheld, EIS off, 1x    -> OIS transfer (what EIS must handle)
  D handheld, EIS off, zoom  -> OIS transfer at zoom
  E handheld, EIS on,  1x    -> end-to-end     PASS: residual <= 0.4x C in 1-8Hz
  F handheld, EIS on,  zoom  -> end-to-end     PASS: residual <= 0.4x D in 1-8Hz

Usage:
  .venv/bin/python eis_matrix.py A=<session_dir> B=<dir> C=<dir> D=<dir> E=<dir> F=<dir>
Any subset of roles is allowed; comparisons print only when both sides exist.
"""
import sys
from pathlib import Path

import numpy as np

from eis_progress import BANDS, ANALYSIS_H, gyro_input, video_residual, telemetry


def fmt_bands(d, fmt="{:.2f}"):
    return "  ".join(f"{b[0]}-{b[1]}Hz=" + fmt.format(d[b]) for b in BANDS)


def main():
    sessions = {}
    for arg in sys.argv[1:]:
        role, _, path = arg.partition("=")
        sessions[role.upper()] = Path(path)

    data = {}
    for role, path in sorted(sessions.items()):
        videos = sorted(path.glob("*.mp4"))
        if not videos:
            print(f"{role}: no video in {path}, skipping")
            continue
        gin, gin_rms = gyro_input(path)
        vres, vres_rms, good, total, fps = video_residual(videos[0])
        tel = telemetry(path)
        data[role] = dict(gin=gin, gin_rms=gin_rms, vres=vres, vres_rms=vres_rms, tel=tel, fps=fps)
        zoom = tel.get("zoom_med", 1.0)
        print(f"[{role}] {path.name}: shake {gin_rms:.1f} deg/s, residual RMS {vres_rms:.2f}px, "
              f"zoom {zoom:.2f}, leash {tel.get('leash_pct', 0):.0f}%")
        print(f"     residual bands: {fmt_bands(vres)}")

    print()
    if "A" in data and "B" in data:
        print("INJECTION TEST (B vs A, braced — does EIS add motion?):")
        ok = True
        for b in BANDS:
            ratio = data["B"]["vres"][b] / max(data["A"]["vres"][b], 1e-9)
            flag = "OK" if ratio <= 1.1 else "FAIL"
            ok &= flag == "OK"
            print(f"  {b[0]}-{b[1]}Hz: {ratio:.2f}x {flag}")
        print(f"  => {'PASS' if ok else 'FAIL'}")
        print()

    for role, label in (("C", "1x"), ("D", "zoom")):
        if role not in data:
            continue
        d = data[role]
        zoom = d["tel"].get("zoom_med", 1.0)
        fx_uv = 0.685
        pred = {b: np.radians(d["gin"][b]) / d["fps"] * fx_uv * zoom * ANALYSIS_H for b in BANDS}
        surv = {b: d["vres"][b] / max(pred[b], 1e-9) for b in BANDS}
        print(f"OIS TRANSFER ({role}, handheld EIS-off {label}): fraction of gyro-predicted "
              f"rotation surviving to screen (the band EIS must handle):")
        print(f"  {fmt_bands(surv)}")
        print()

    for on, off, label in (("E", "C", "1x"), ("F", "D", "zoom")):
        if on not in data or off not in data:
            continue
        # normalize by shake input: takes differ, so compare suppression not raw residual
        print(f"END-TO-END ({on} vs {off}, {label}): residual ratio per band, "
              f"shake-normalized (x{data[off]['gin_rms'] / data[on]['gin_rms']:.2f}):")
        norm = data[off]["gin_rms"] / max(data[on]["gin_rms"], 1e-9)
        ok = True
        for b in BANDS:
            ratio = (data[on]["vres"][b] * norm) / max(data[off]["vres"][b], 1e-9)
            target = 0.4 if b[0] >= 1 and b[1] <= 8 else None
            flag = "" if target is None else (" OK" if ratio <= target else " FAIL")
            if target is not None:
                ok &= ratio <= target
            print(f"  {b[0]}-{b[1]}Hz: {ratio:.2f}x{flag}")
        print(f"  => 1-8Hz target <=0.40: {'PASS' if ok else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
