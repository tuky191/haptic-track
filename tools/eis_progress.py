#!/usr/bin/env python3
"""EIS progress scorecard — empirical, per-session, build-comparable.

Measures (a) hand-shake input from gyro_raw.csv, (b) residual on-screen
jitter from the recorded video via phase correlation, and reports per-band
suppression (output/input). Appends one row per session to a ledger CSV so
stabilization progress across builds is a table, not a gut feeling.

Protocol for comparable numbers: same scene, same approximate distance and
zoom, ~20s handheld hold, EIS on. Run:
    .venv/bin/python eis_progress.py ../bench_data/s26_eis/session_<ts> [--label "build/desc"]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

BANDS = [(0.2, 1), (1, 3), (3, 8), (8, 15)]
ANALYSIS_H, ANALYSIS_W = 480, 270  # portrait 4K downscale


def band_rms(sig, fs):
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    f = np.fft.rfftfreq(len(sig), 1 / fs)
    p = np.abs(np.fft.rfft(sig)) ** 2 / len(sig)
    return {b: float(np.sqrt(p[(f >= b[0]) & (f < b[1])].sum())) for b in BANDS}


def gyro_input(session: Path):
    """Per-band angular-rate shake (deg-equivalent displacement per frame at 30fps)."""
    g = np.loadtxt(session / "gyro_raw.csv", delimiter=",", skiprows=1)
    t = (g[:, 0] - g[0, 0]) / 1e9
    q = g[:, 1:5]
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
    dang = np.degrees(2 * np.arccos(dots))
    dt = np.diff(t)
    ok = dt > 1e-6
    w = dang[ok] / dt[ok]
    fs = 1 / np.median(dt)
    tu = np.arange(t[0], t[-1], 1 / fs)
    wu = np.interp(tu, t[1:][ok], w)
    return band_rms(wu, fs), float(np.sqrt(np.mean(wu**2)))


def video_residual(video: Path, max_frames=1200):
    """Per-band residual jitter (px at 480p) from phase correlation, response-gated."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hann = None
    prev = None
    disp = []
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (ANALYSIS_W, ANALYSIS_H)).astype(np.float32)
        crop = gray[60:420, 35:235]
        if hann is None:
            hann = cv2.createHanningWindow((crop.shape[1], crop.shape[0]), cv2.CV_32F)
        crop = crop * hann
        if prev is not None:
            (du, dv), resp = cv2.phaseCorrelate(prev, crop)
            disp.append((du, dv, resp))
        prev = crop
        n += 1
    cap.release()
    d = np.array(disp)
    good = d[:, 2] > max(0.05, np.percentile(d[:, 2], 20))
    du, dv = d[:, 0].copy(), d[:, 1].copy()
    du[~good] = 0.0
    dv[~good] = 0.0
    mag = np.hypot(du, dv)
    bu, bv = band_rms(du, fps), band_rms(dv, fps)
    bands = {b: float(np.hypot(bu[b], bv[b])) for b in BANDS}
    return bands, float(np.sqrt(np.mean(mag**2))), int(good.sum()), len(d), fps


def telemetry(session: Path):
    f = session / "corrections.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return {}
    out = {
        "frames": len(rows),
        "leash_pct": 100.0 * sum(int(r["leash"]) for r in rows) / len(rows),
        "corr_deg_mean": float(np.mean([float(r["corr_deg"]) for r in rows])),
    }
    if "zoom" in rows[0]:
        zooms = [float(r["zoom"]) for r in rows]
        out["zoom_med"] = float(np.median(zooms))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--ledger", type=Path, default=None)
    args = ap.parse_args()

    session = args.session
    videos = sorted(session.glob("*.mp4"))
    if not videos:
        sys.exit(f"no .mp4 in {session}")
    video = videos[0]

    gin, gin_rms = gyro_input(session)
    vres, vres_rms, good, total, fps = video_residual(video)
    tel = telemetry(session)
    zoom = tel.get("zoom_med", 1.0)

    # Suppression: residual px normalized by gyro-predicted px for the same band.
    # predicted px/frame ~ rad/s * (1/fps) * fx_uv * zoom * frame_height
    fx_uv = 0.685
    pred = {b: np.radians(gin[b]) / fps * fx_uv * zoom * ANALYSIS_H for b in BANDS}
    supp = {b: (vres[b] / pred[b] if pred[b] > 1e-9 else float("nan")) for b in BANDS}

    print(f"session: {session.name}  label: {args.label or '-'}")
    print(f"  input shake (gyro): RMS {gin_rms:.1f} deg/s | bands "
          + " ".join(f"{b[0]}-{b[1]}Hz={gin[b]:.2f}" for b in BANDS))
    print(f"  residual (video):   RMS {vres_rms:.2f} px@480p | bands "
          + " ".join(f"{b[0]}-{b[1]}Hz={vres[b]:.2f}" for b in BANDS))
    print(f"  residual/input ratio per band (lower=better, >1 = motion beyond gyro [translation]):")
    print("    " + "  ".join(f"{b[0]}-{b[1]}Hz={supp[b]:.2f}" for b in BANDS))
    if tel:
        print(f"  telemetry: zoom_med={zoom:.2f} leash={tel.get('leash_pct', 0):.0f}% "
              f"corr_mean={tel.get('corr_deg_mean', 0):.2f}deg  phase-corr good {good}/{total}")

    ledger = args.ledger or session.parent / "eis_progress.csv"
    new = not ledger.exists()
    with open(ledger, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["session", "label", "gyro_rms_dps", "video_rms_px", "zoom_med", "leash_pct"]
                       + [f"in_{b[0]}_{b[1]}" for b in BANDS]
                       + [f"res_{b[0]}_{b[1]}" for b in BANDS]
                       + [f"ratio_{b[0]}_{b[1]}" for b in BANDS])
        w.writerow([session.name, args.label, f"{gin_rms:.2f}", f"{vres_rms:.3f}",
                    f"{zoom:.2f}", f"{tel.get('leash_pct', 0):.1f}"]
                   + [f"{gin[b]:.3f}" for b in BANDS]
                   + [f"{vres[b]:.3f}" for b in BANDS]
                   + [f"{supp[b]:.3f}" for b in BANDS])
    print(f"  → appended to {ledger}")


if __name__ == "__main__":
    main()
