#!/usr/bin/env python3
"""Phase 0: validate the eis_progress measurement chain against known ground truth.

Measures a real capture twice — as-is, and with a KNOWN synthetic displacement
(sum of sinusoids) warped onto each frame — through the exact phase-correlation
band measurement used by eis_progress.py. The injected signal must be recovered
as the band-power difference (injected and real motion are independent).

Usage: .venv/bin/python validate_meter.py <video.mp4>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

BANDS = [(0.2, 1), (1, 3), (3, 8), (8, 15)]
FPS = 30.0
N_FRAMES = 600

# injected shake: (freq Hz, amplitude px at 480p) per axis
INJECT_U = [(2.0, 6.0), (10.0, 2.0)]
INJECT_V = [(5.0, 4.0)]


def injected_series(comps, n):
    t = np.arange(n) / FPS
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in comps)


def band_power(sig, fs):
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    sig = sig * np.hanning(len(sig))  # window the series — raw FFT leaks across bands
    f = np.fft.rfftfreq(len(sig), 1 / fs)
    p = np.abs(np.fft.rfft(sig)) ** 2 / len(sig)
    return {b: float(p[(f >= b[0]) & (f < b[1])].sum()) for b in BANDS}


def measure(video, pos_u=None, pos_v=None):
    cap = cv2.VideoCapture(str(video))
    hann = None
    prev = None
    meas = []
    i = 0
    while i < N_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (270, 480)).astype(np.float32)
        if pos_u is not None:
            m = np.float32([[1, 0, pos_u[i]], [0, 1, pos_v[i]]])
            g = cv2.warpAffine(g, m, (270, 480), borderMode=cv2.BORDER_REPLICATE)
        crop = g[60:420, 35:235]
        if hann is None:
            hann = cv2.createHanningWindow((crop.shape[1], crop.shape[0]), cv2.CV_32F)
        crop = crop * hann
        if prev is not None:
            (du, dv), resp = cv2.phaseCorrelate(prev, crop)
            meas.append((du, dv, resp))
        prev = crop
        i += 1
    cap.release()
    return np.array(meas)


def main():
    video = Path(sys.argv[1])
    pos_u = injected_series(INJECT_U, N_FRAMES)
    pos_v = injected_series(INJECT_V, N_FRAMES)

    base = measure(video)
    warped = measure(video, pos_u, pos_v)
    n = min(len(base), len(warped))
    gt_u = np.diff(pos_u[: n + 1])
    gt_v = np.diff(pos_v[: n + 1])

    print(f"frames: {n}, response med base={np.median(base[:,2]):.3f} warped={np.median(warped[:,2]):.3f}")
    ok = True
    for axis, col, gt in (("u", 0, gt_u), ("v", 1, gt_v)):
        pb = band_power(base[:n, col], FPS)
        pw = band_power(warped[:n, col], FPS)
        pg = band_power(gt, FPS)
        print(f"  {axis}-axis (recovered vs injected, band-power difference):")
        for b in BANDS:
            if pg[b] < 0.01:
                continue
            rec = np.sqrt(max(pw[b] - pb[b], 0.0))
            inj = np.sqrt(pg[b])
            ratio = rec / inj
            flag = "OK" if 0.85 <= ratio <= 1.15 else "FAIL"
            if flag == "FAIL":
                ok = False
            print(f"    {b[0]}-{b[1]}Hz: injected={inj:.3f}px recovered={rec:.3f}px ratio={ratio:.3f} {flag}")
    print("\nMETER VALID" if ok else "\nMETER INVALID — do not trust band numbers")


if __name__ == "__main__":
    main()
