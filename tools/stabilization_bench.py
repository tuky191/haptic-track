#!/usr/bin/env python3
"""
Off-device bench test for gyro-based EIS (Electronic Image Stabilization).

Loads raw video + gyro sensor data captured on device, replays the stabilization
algorithm, measures quality via phase correlation, and compares against a non-causal
"ideal" reference.

Usage:
    python stabilization_bench.py <bench_dir> <raw_video>

    bench_dir/  — adb pull from /sdcard/Android/data/com.haptictrack/files/bench/session_*
      gyro_raw.csv       — timestamp_ns, w, x, y, z  (~200 Hz)
      frames.csv         — frame_idx, timestamp_ns
      bench_params.csv   — timeConstant, cropZoom, fxUv, fyUv, clampMarginFraction

    raw_video   — the .mp4 recorded at the same time

Outputs (in bench_dir/results/):
    metrics.txt          — summary table
    displacement.png     — frame-to-frame displacement time series
    spectrum.png         — frequency spectrum of jitter
    correction.png       — causal vs non-causal correction trajectory
    alignment.png        — gyro-predicted vs optical-flow scatter
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Quaternion math  — exact port of GyroStabilizer.kt
# ---------------------------------------------------------------------------

def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else q

def quat_multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_dot(a, b):
    return np.dot(a, b)

def slerp(a, b, t):
    dot = quat_dot(a, b)
    if dot < 0:
        dot = -dot
        b2 = -b
    else:
        b2 = b

    if dot > 0.9995:
        result = a + t * (b2 - a)
        return quat_normalize(result)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    wa = np.sin((1 - t) * theta) / sin_theta
    wb = np.sin(t * theta) / sin_theta
    return quat_normalize(wa * a + wb * b2)

def quat_to_rotation_matrix(q):
    w, x, y, z = q
    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),     2*(xz+wy)],
        [    2*(xy+wz), 1 - 2*(xx+zz),     2*(yz-wx)],
        [    2*(xz-wy),     2*(yz+wx), 1 - 2*(xx+yy)],
    ])

def compute_homography_uv(R, fx, fy, zoom):
    """H = S × K × R × K⁻¹  in UV [0,1]² space. Returns 3×3 row-major."""
    # K × R
    kr = np.array([
        [fx*R[0,0] + 0.5*R[2,0],  fx*R[0,1] + 0.5*R[2,1],  fx*R[0,2] + 0.5*R[2,2]],
        [fy*R[1,0] + 0.5*R[2,0],  fy*R[1,1] + 0.5*R[2,1],  fy*R[1,2] + 0.5*R[2,2]],
        [               R[2,0],                R[2,1],                R[2,2]],
    ])
    # (K × R) × K⁻¹
    ifx = 1.0 / fx
    ify = 1.0 / fy
    h = np.array([
        [kr[0,0]*ifx,  kr[0,1]*ify,  kr[0,2] - kr[0,0]*0.5*ifx - kr[0,1]*0.5*ify],
        [kr[1,0]*ifx,  kr[1,1]*ify,  kr[1,2] - kr[1,0]*0.5*ifx - kr[1,1]*0.5*ify],
        [kr[2,0]*ifx,  kr[2,1]*ify,  kr[2,2] - kr[2,0]*0.5*ifx - kr[2,1]*0.5*ify],
    ])
    # Crop zoom: scale inward
    iz = 1.0 / zoom
    tx = 0.5 * (1.0 - iz)
    h[0, :] = iz * h[0, :];  h[0, 2] += tx
    h[1, :] = iz * h[1, :];  h[1, 2] += tx
    return h

def max_corner_excursion(H):
    """Max OOB distance of any corner under homography H (row-major 3×3)."""
    max_exc = 0.0
    for u in (0.0, 1.0):
        for v in (0.0, 1.0):
            tu = H[0,0]*u + H[0,1]*v + H[0,2]
            tv = H[1,0]*u + H[1,1]*v + H[1,2]
            exc = max(-tu, tu - 1, -tv, tv - 1, 0)
            max_exc = max(max_exc, exc)
    return max_exc

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class BenchParams:
    time_constant: float
    crop_zoom: float
    fx_uv: float
    fy_uv: float
    clamp_margin_fraction: float

def load_params(bench_dir: Path) -> BenchParams:
    with open(bench_dir / "bench_params.csv") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return BenchParams(
        time_constant=float(row["timeConstant"]),
        crop_zoom=float(row["cropZoom"]),
        fx_uv=float(row["fxUv"]),
        fy_uv=float(row["fyUv"]),
        clamp_margin_fraction=float(row["clampMarginFraction"]),
    )

def load_gyro(bench_dir: Path):
    """Returns (timestamps_ns, quaternions) as numpy arrays."""
    data = np.loadtxt(bench_dir / "gyro_raw.csv", delimiter=",", skiprows=1)
    return data[:, 0].astype(np.int64), data[:, 1:5]

def load_frame_timestamps(bench_dir: Path):
    """Returns (frame_indices, timestamps_ns) as numpy arrays."""
    data = np.loadtxt(bench_dir / "frames.csv", delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0].astype(np.int64), data[:, 1].astype(np.int64)

# ---------------------------------------------------------------------------
# Algorithm replay  — exact match to GyroStabilizer.onSensorChanged
# ---------------------------------------------------------------------------

def replay_stabilization(gyro_ts, gyro_quats, params: BenchParams, sensor_orientation=90):
    """
    Replays the stabilization algorithm on gyro data.
    Returns per-sample correction homographies (3×3 row-major, N×3×3).
    Also returns the smoothed quaternion trajectory.
    """
    angle = np.radians(sensor_orientation)
    d2s_quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    n = len(gyro_ts)
    corrections_h = np.zeros((n, 3, 3))
    smoothed_quats = np.zeros((n, 4))
    identity = np.eye(3)

    SENSOR_GAP_NS = 100_000_000

    smoothed = gyro_quats[0].copy()
    sample_rate = 200.0
    initialized = False
    last_ts = gyro_ts[0]

    for i in range(n):
        raw = quat_normalize(gyro_quats[i])
        now_ns = gyro_ts[i]

        if not initialized:
            smoothed = raw.copy()
            initialized = True
            last_ts = now_ns
            smoothed_quats[i] = smoothed
            corrections_h[i] = identity
            continue

        dt_ns = now_ns - last_ts
        last_ts = now_ns
        if dt_ns <= 0:
            smoothed_quats[i] = smoothed
            corrections_h[i] = corrections_h[max(0, i-1)]
            continue
        if dt_ns > SENSOR_GAP_NS:
            smoothed = raw.copy()
            smoothed_quats[i] = smoothed
            corrections_h[i] = identity
            continue

        dt_sec = dt_ns / 1e9
        sample_rate = 0.95 * sample_rate + 0.05 * (1.0 / dt_sec)

        alpha = 1.0 - np.exp(-(1.0 / sample_rate) / params.time_constant)
        smoothed = slerp(smoothed, raw, alpha)

        # Leash: limit deviation between smoothed and raw so the correction
        # always fits within the crop margin (replaces the old hard clamp)
        crop_margin = 0.5 * (1.0 - 1.0 / params.crop_zoom)
        max_corr_angle = crop_margin / max(params.fx_uv, params.fy_uv)
        dev_quat = quat_multiply(quat_conjugate(smoothed), raw)
        dev_angle = 2.0 * np.arccos(np.clip(dev_quat[0], -1.0, 1.0))
        if dev_angle > max_corr_angle and dev_angle > 1e-6:
            catch_up = 1.0 - max_corr_angle / dev_angle
            smoothed = slerp(smoothed, raw, catch_up)

        smoothed_quats[i] = smoothed

        correction_device = quat_multiply(quat_conjugate(smoothed), raw)
        correction = quat_multiply(quat_multiply(d2s_quat, correction_device),
                                   quat_conjugate(d2s_quat))
        R = quat_to_rotation_matrix(correction)
        h = compute_homography_uv(R, params.fx_uv, params.fy_uv, params.crop_zoom)

        corrections_h[i] = h

    return corrections_h, smoothed_quats

def replay_noncausal(gyro_ts, gyro_quats, params: BenchParams, sensor_orientation=90):
    """
    Non-causal (bidirectional) smoothing — the theoretical ideal.
    Forward SLERP + backward SLERP, averaged.
    """
    n = len(gyro_ts)
    SENSOR_GAP_NS = 100_000_000

    def run_pass(ts, quats, sample_rate_init=200.0):
        smoothed_out = np.zeros_like(quats)
        smoothed = quat_normalize(quats[0])
        smoothed_out[0] = smoothed
        sample_rate = sample_rate_init
        last_ts = ts[0]
        for i in range(1, len(ts)):
            raw = quat_normalize(quats[i])
            dt_ns = ts[i] - last_ts
            last_ts = ts[i]
            if dt_ns <= 0 or dt_ns > SENSOR_GAP_NS:
                smoothed = raw.copy()
                smoothed_out[i] = smoothed
                continue
            dt_sec = dt_ns / 1e9
            sample_rate = 0.95 * sample_rate + 0.05 * (1.0 / dt_sec)
            alpha = 1.0 - np.exp(-(1.0 / sample_rate) / params.time_constant)
            smoothed = slerp(smoothed, raw, alpha)
            smoothed_out[i] = smoothed
        return smoothed_out

    # Forward pass
    fwd = run_pass(gyro_ts, gyro_quats)

    # Backward pass (reverse time)
    rev_ts = gyro_ts[::-1].copy()
    rev_ts = rev_ts[0] - (rev_ts - rev_ts[-1])  # make timestamps ascending
    rev_quats = gyro_quats[::-1].copy()
    bwd_rev = run_pass(rev_ts, rev_quats)
    bwd = bwd_rev[::-1]

    # Average forward and backward via SLERP(fwd, bwd, 0.5)
    ideal_smoothed = np.zeros_like(gyro_quats)
    for i in range(n):
        ideal_smoothed[i] = slerp(fwd[i], bwd[i], 0.5)

    # Compute corrections from ideal smoothing
    angle = np.radians(sensor_orientation)
    d2s_quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    corrections_h = np.zeros((n, 3, 3))
    for i in range(n):
        raw = quat_normalize(gyro_quats[i])
        correction_device = quat_multiply(quat_conjugate(ideal_smoothed[i]), raw)
        correction = quat_multiply(quat_multiply(d2s_quat, correction_device),
                                   quat_conjugate(d2s_quat))
        R = quat_to_rotation_matrix(correction)
        h = compute_homography_uv(R, params.fx_uv, params.fy_uv, params.crop_zoom)

        excursion = max_corner_excursion(h)
        crop_margin = 0.5 * (1.0 - 1.0 / params.crop_zoom)
        usable_margin = crop_margin * params.clamp_margin_fraction
        if excursion > usable_margin:
            clamp_ratio = usable_margin / excursion
            clamped_q = slerp(np.array([1.0, 0, 0, 0]), correction, clamp_ratio)
            R_c = quat_to_rotation_matrix(clamped_q)
            h = compute_homography_uv(R_c, params.fx_uv, params.fy_uv, params.crop_zoom)

        corrections_h[i] = h

    return corrections_h, ideal_smoothed

# ---------------------------------------------------------------------------
# Video displacement measurement
# ---------------------------------------------------------------------------

def measure_frame_displacements(video_path: str):
    """Phase correlation between consecutive frames → (dx, dy, response) in pixels."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps")

    prev_gray = None
    displacements = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if prev_gray is not None:
            (dx, dy), response = cv2.phaseCorrelate(prev_gray, gray)
            displacements.append((dx, dy, response))
        prev_gray = gray
        frame_count += 1

    cap.release()
    responses = np.array([d[2] for d in displacements])
    good = (responses > 0.1).sum()
    print(f"Measured {len(displacements)} frame-pair displacements from {frame_count} frames")
    print(f"  Phase correlation: median_response={np.median(responses):.3f}, "
          f"good (>0.1): {good}/{len(displacements)}")
    return np.array(displacements), fps, width, height

# ---------------------------------------------------------------------------
# Correction → displacement conversion
# ---------------------------------------------------------------------------

def homography_translation(H):
    """Extract (tx, ty) translation from a homography in UV space."""
    # At center (0.5, 0.5): output = H @ [0.5, 0.5, 1]
    cx = H[0,0]*0.5 + H[0,1]*0.5 + H[0,2]
    cy = H[1,0]*0.5 + H[1,1]*0.5 + H[1,2]
    return cx - 0.5, cy - 0.5

def corrections_to_frame_displacements(corrections_h, gyro_ts, frame_ts, width, height):
    """
    Interpolate gyro corrections to video frame times and convert to pixel displacements.
    Returns (N_frames-1, 2) array of predicted inter-frame displacement in pixels.
    """
    n_frames = len(frame_ts)
    frame_disp = []
    for i in range(n_frames - 1):
        # Find gyro correction at each frame time
        idx0 = np.searchsorted(gyro_ts, frame_ts[i], side='right') - 1
        idx1 = np.searchsorted(gyro_ts, frame_ts[i+1], side='right') - 1
        idx0 = np.clip(idx0, 0, len(corrections_h) - 1)
        idx1 = np.clip(idx1, 0, len(corrections_h) - 1)

        tx0, ty0 = homography_translation(corrections_h[idx0])
        tx1, ty1 = homography_translation(corrections_h[idx1])

        # Inter-frame correction change → portrait video pixels
        # homography_translation returns (tx, ty) in sensor UV; portrait swaps u→y, v→x
        dpx = (ty1 - ty0) * width   # sensor v → portrait x
        dpy = (tx1 - tx0) * height  # sensor u → portrait y
        frame_disp.append((dpx, dpy))

    return np.array(frame_disp)


def raw_gyro_frame_displacements(gyro_ts, gyro_quats, frame_ts, params, sensor_orientation=90):
    """
    Compute raw gyro inter-frame rotation projected to pixel displacement.
    This is the total camera motion between frames — should correlate with optical flow.
    """
    angle = np.radians(sensor_orientation)
    d2s_quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])

    n_frames = len(frame_ts)
    frame_disp = []
    for i in range(n_frames - 1):
        idx0 = np.searchsorted(gyro_ts, frame_ts[i], side='right') - 1
        idx1 = np.searchsorted(gyro_ts, frame_ts[i+1], side='right') - 1
        idx0 = np.clip(idx0, 0, len(gyro_quats) - 1)
        idx1 = np.clip(idx1, 0, len(gyro_quats) - 1)

        q0 = quat_normalize(gyro_quats[idx0])
        q1 = quat_normalize(gyro_quats[idx1])

        # Relative rotation in device space: how the device moved between frames
        q_rel_device = quat_multiply(quat_conjugate(q0), q1)
        # Rotate to sensor space
        q_rel_sensor = quat_multiply(quat_multiply(d2s_quat, q_rel_device),
                                     quat_conjugate(d2s_quat))

        R = quat_to_rotation_matrix(q_rel_sensor)
        # Project center pixel through rotation to get displacement
        # In UV space: center (0.5, 0.5) → R × K⁻¹ [0.5, 0.5, 1] → K × result
        # Simplified: displacement ≈ (R[0,2]*fx, R[1,2]*fy) for small rotations
        # Full projection for accuracy:
        cx_in = 0.5
        cy_in = 0.5
        # K⁻¹ [cx, cy, 1]
        px = (cx_in - 0.5) / params.fx_uv  # = 0
        py = (cy_in - 0.5) / params.fy_uv  # = 0
        pz = 1.0
        # R × [px, py, pz]
        rx = R[0,0]*px + R[0,1]*py + R[0,2]*pz
        ry = R[1,0]*px + R[1,1]*py + R[1,2]*pz
        rz = R[2,0]*px + R[2,1]*py + R[2,2]*pz
        # K × [rx, ry, rz] → UV
        out_u = params.fx_uv * (rx / rz) + 0.5
        out_v = params.fy_uv * (ry / rz) + 0.5
        # Displacement in sensor UV → portrait video: sensor v → pixel x, sensor u → pixel y
        du = out_u - cx_in
        dv = out_v - cy_in
        frame_disp.append((dv, du))

    return np.array(frame_disp)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(raw_disp, causal_pred, noncausal_pred, gyro_raw_disp, fps, width, height):
    """Compute quality metrics."""
    raw_dx, raw_dy = raw_disp[:, 0], raw_disp[:, 1]
    raw_mag = np.sqrt(raw_dx**2 + raw_dy**2)
    raw_jitter_rms = np.sqrt(np.mean(raw_mag**2))

    # What the stabilized video's residual would be:
    # residual = raw displacement - correction displacement
    n = min(len(raw_disp), len(causal_pred))
    residual_causal = raw_disp[:n, :2] - causal_pred[:n]
    residual_nc = raw_disp[:n, :2] - noncausal_pred[:n]

    causal_mag = np.sqrt(residual_causal[:, 0]**2 + residual_causal[:, 1]**2)
    nc_mag = np.sqrt(residual_nc[:, 0]**2 + residual_nc[:, 1]**2)

    causal_rms = np.sqrt(np.mean(causal_mag**2))
    nc_rms = np.sqrt(np.mean(nc_mag**2))

    def safe_corr(a, b):
        return np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else float('nan')

    # Gyro-video correlation using high-confidence phase correlation frames only
    n_corr = min(len(raw_disp), len(gyro_raw_disp))
    gyro_raw_px = gyro_raw_disp[:n_corr] * np.array([width, height])
    good_mask = raw_disp[:n_corr, 2] > 0.1  # phase correlation response > 0.1
    n_good = good_mask.sum()
    if n_good > 5:
        corr_x = safe_corr(raw_disp[:n_corr, 0][good_mask], gyro_raw_px[:n_corr, 0][good_mask])
        corr_y = safe_corr(raw_disp[:n_corr, 1][good_mask], gyro_raw_px[:n_corr, 1][good_mask])
    else:
        corr_x = corr_y = float('nan')

    gyro_raw_rms = np.sqrt(np.mean(gyro_raw_px[:n_corr, 0]**2 + gyro_raw_px[:n_corr, 1]**2))
    if n_good > 5:
        optical_rms_good = np.sqrt(np.mean(raw_disp[:n_corr, 0][good_mask]**2 +
                                            raw_disp[:n_corr, 1][good_mask]**2))
        gyro_rms_good = np.sqrt(np.mean(gyro_raw_px[:n_corr, 0][good_mask]**2 +
                                         gyro_raw_px[:n_corr, 1][good_mask]**2))
        scale_ratio = optical_rms_good / gyro_rms_good if gyro_rms_good > 0 else float('inf')
    else:
        scale_ratio = raw_jitter_rms / gyro_raw_rms if gyro_raw_rms > 0 else float('inf')

    # Correction tracking error (causal vs noncausal)
    track_err = np.sqrt(np.mean((causal_pred[:n] - noncausal_pred[:n])**2))

    return {
        "n_good_frames": int(n_good),
        "raw_jitter_rms_px": raw_jitter_rms,
        "gyro_raw_rms_px": gyro_raw_rms,
        "scale_ratio": scale_ratio,
        "causal_residual_rms_px": causal_rms,
        "noncausal_residual_rms_px": nc_rms,
        "improvement_ratio": raw_jitter_rms / causal_rms if causal_rms > 0 else float('inf'),
        "ideal_improvement_ratio": raw_jitter_rms / nc_rms if nc_rms > 0 else float('inf'),
        "gyro_video_corr_x": corr_x,
        "gyro_video_corr_y": corr_y,
        "causal_vs_ideal_error_px": track_err,
        "n_frames": n,
        "fps": fps,
    }

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_plots(raw_disp, causal_pred, noncausal_pred, gyro_raw_disp, metrics,
                    out_dir: Path, fps, width, height):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    n = metrics["n_frames"]
    t = np.arange(n) / fps  # seconds
    gyro_raw_px = gyro_raw_disp[:n] * np.array([width, height])

    # 1. Displacement time series — raw gyro vs optical flow
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, raw_disp[:n, 0], alpha=0.6, label="Optical flow dx")
    axes[0].plot(t, gyro_raw_px[:, 0], alpha=0.8, label="Gyro raw dx")
    axes[0].set_ylabel("Displacement X (px)")
    axes[0].legend()
    axes[0].set_title("Frame-to-frame displacement: X axis")

    axes[1].plot(t, raw_disp[:n, 1], alpha=0.6, label="Optical flow dy")
    axes[1].plot(t, gyro_raw_px[:, 1], alpha=0.8, label="Gyro raw dy")
    axes[1].set_ylabel("Displacement Y (px)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].set_title("Frame-to-frame displacement: Y axis")
    plt.tight_layout()
    plt.savefig(out_dir / "displacement.png", dpi=150)
    plt.close()

    # 2. Residual jitter
    residual_causal = raw_disp[:n, :2] - causal_pred[:n]
    residual_nc = raw_disp[:n, :2] - noncausal_pred[:n]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax_idx, (label, axis) in enumerate([("X", 0), ("Y", 1)]):
        axes[ax_idx].plot(t, raw_disp[:n, axis], alpha=0.4, label=f"Raw jitter {label}")
        axes[ax_idx].plot(t, residual_causal[:, axis], alpha=0.7, label=f"Residual (causal) {label}")
        axes[ax_idx].plot(t, residual_nc[:, axis], alpha=0.5, ls="--", label=f"Residual (ideal) {label}")
        axes[ax_idx].set_ylabel(f"Jitter {label} (px)")
        axes[ax_idx].legend()
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title("Residual jitter after stabilization")
    plt.tight_layout()
    plt.savefig(out_dir / "residual.png", dpi=150)
    plt.close()

    # 3. Frequency spectrum
    if n > 16:
        from scipy.fft import rfft, rfftfreq
        freq = rfftfreq(n, d=1.0/fps)
        raw_mag_x = np.abs(rfft(raw_disp[:n, 0]))
        causal_mag_x = np.abs(rfft(residual_causal[:, 0]))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.semilogy(freq[1:], raw_mag_x[1:], alpha=0.6, label="Raw jitter X")
        ax.semilogy(freq[1:], causal_mag_x[1:], alpha=0.8, label="Residual (causal) X")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Jitter frequency spectrum")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "spectrum.png", dpi=150)
        plt.close()

    # 4. Gyro vs optical alignment scatter (raw gyro, not correction)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(raw_disp[:n, 0], gyro_raw_px[:, 0], alpha=0.3, s=8)
    axes[0].set_xlabel("Optical dx (px)")
    axes[0].set_ylabel("Gyro raw dx (px)")
    axes[0].set_title(f"X alignment (r={metrics['gyro_video_corr_x']:.3f})")
    lim = max(abs(raw_disp[:n, 0]).max(), abs(gyro_raw_px[:, 0]).max()) * 1.1
    axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
    axes[0].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5)

    axes[1].scatter(raw_disp[:n, 1], gyro_raw_px[:, 1], alpha=0.3, s=8)
    axes[1].set_xlabel("Optical dy (px)")
    axes[1].set_ylabel("Gyro raw dy (px)")
    axes[1].set_title(f"Y alignment (r={metrics['gyro_video_corr_y']:.3f})")
    lim = max(abs(raw_disp[:n, 1]).max(), abs(gyro_raw_px[:, 1]).max()) * 1.1
    axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    axes[1].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5)

    plt.suptitle("Raw gyro vs. optical-flow displacement (scale_ratio={:.2f})".format(
        metrics['scale_ratio']))
    plt.tight_layout()
    plt.savefig(out_dir / "alignment.png", dpi=150)
    plt.close()

    print(f"Plots saved to {out_dir}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sensor_uv_to_portrait_px(H_uv, portrait_w, portrait_h):
    """Convert a sensor-UV homography to portrait-pixel homography.

    CameraX encodes portrait frames by rotating sensor landscape 90° CCW.
    Coordinate mapping: portrait_pu = sensor_v, portrait_pv = 1 - sensor_u.
    So sensor u = 1 - pv, sensor v = pu.
    """
    H = H_uv
    W, Hgt = float(portrait_w), float(portrait_h)
    return np.array([
        [H[1,1],           -H[1,0]*W/Hgt,    (H[1,0] + H[1,2])*W              ],
        [-H[0,1]*Hgt/W,     H[0,0],           (1.0 - H[0,0] - H[0,2])*Hgt     ],
        [H[2,1]/W,          -H[2,0]/Hgt,       H[2,0] + H[2,2]                 ],
    ])


def generate_stabilized_video(input_path: str, output_path: str, corrections_h,
                               gyro_ts, frame_ts):
    """Apply per-frame homographies to raw video and write stabilized output."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(len(frame_ts)):
        ret, frame = cap.read()
        if not ret:
            break
        idx = np.searchsorted(gyro_ts, frame_ts[i], side='right') - 1
        idx = np.clip(idx, 0, len(corrections_h) - 1)
        H_uv = corrections_h[idx]
        H_px = sensor_uv_to_portrait_px(H_uv, width, height)
        stabilized = cv2.warpPerspective(frame, H_px, (width, height),
                                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_REPLICATE)
        writer.write(stabilized)

    cap.release()
    writer.release()
    print(f"Stabilized video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EIS stabilization bench test")
    parser.add_argument("bench_dir", help="Directory with gyro_raw.csv, frames.csv, bench_params.csv")
    parser.add_argument("raw_video", help="Raw .mp4 video recorded at the same time")
    parser.add_argument("--sensor-orientation", type=int, default=90,
                        help="SENSOR_ORIENTATION degrees (default: 90)")
    parser.add_argument("--output-video", action="store_true",
                        help="Generate stabilized output video")
    parser.add_argument("--tc", type=float, default=None,
                        help="Override time constant (for parameter sweep)")
    parser.add_argument("--crop", type=float, default=None,
                        help="Override crop zoom (for parameter sweep)")
    parser.add_argument("--fx", type=float, default=None,
                        help="Override fx_uv focal length")
    parser.add_argument("--fy", type=float, default=None,
                        help="Override fy_uv focal length")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    out_dir = bench_dir / "results"
    out_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading bench data...")
    params = load_params(bench_dir)
    if args.tc is not None:
        params.time_constant = args.tc
    if args.crop is not None:
        params.crop_zoom = args.crop
    if args.fx is not None:
        params.fx_uv = args.fx
    if args.fy is not None:
        params.fy_uv = args.fy
    print(f"  Params: tc={params.time_constant}, crop={params.crop_zoom}, "
          f"fx={params.fx_uv:.3f}, fy={params.fy_uv:.3f}, clamp={params.clamp_margin_fraction}")

    gyro_ts, gyro_quats = load_gyro(bench_dir)
    print(f"  Gyro: {len(gyro_ts)} samples, "
          f"{(gyro_ts[-1]-gyro_ts[0])/1e9:.1f}s, "
          f"~{len(gyro_ts)/((gyro_ts[-1]-gyro_ts[0])/1e9):.0f}Hz")

    frame_idx, frame_ts = load_frame_timestamps(bench_dir)
    print(f"  Frames: {len(frame_ts)} timestamps")
    print(f"  Sensor orientation: {args.sensor_orientation}°")

    # Measure raw video displacement
    print("\nMeasuring raw video displacement (phase correlation)...")
    raw_disp, fps, width, height = measure_frame_displacements(args.raw_video)

    # Compute raw gyro inter-frame displacement (for correlation check)
    print("\nComputing raw gyro displacements...")
    gyro_raw_disp = raw_gyro_frame_displacements(
        gyro_ts, gyro_quats, frame_ts, params, args.sensor_orientation)
    print(f"  {len(gyro_raw_disp)} inter-frame gyro displacements")

    # Replay stabilization (causal)
    print("Replaying causal stabilization...")
    causal_h, causal_smoothed = replay_stabilization(
        gyro_ts, gyro_quats, params, args.sensor_orientation)
    causal_pred = corrections_to_frame_displacements(causal_h, gyro_ts, frame_ts, width, height)
    print(f"  {len(causal_pred)} inter-frame predictions")

    # Replay non-causal ideal
    print("Computing non-causal ideal...")
    nc_h, nc_smoothed = replay_noncausal(
        gyro_ts, gyro_quats, params, args.sensor_orientation)
    nc_pred = corrections_to_frame_displacements(nc_h, gyro_ts, frame_ts, width, height)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(raw_disp, causal_pred, nc_pred, gyro_raw_disp, fps, width, height)

    # Axis alignment diagnostic — raw quaternion components, high-confidence frames only
    n_diag = min(len(raw_disp), len(frame_ts) - 1)
    diag_mask = raw_disp[:n_diag, 2] > 0.1
    n_diag_good = diag_mask.sum()
    print(f"\nAxis alignment diagnostic ({n_diag_good} high-confidence frames):")
    raw_qrel = []
    for i in range(n_diag):
        idx0 = np.searchsorted(gyro_ts, frame_ts[i], side='right') - 1
        idx1 = np.searchsorted(gyro_ts, frame_ts[i+1], side='right') - 1
        idx0, idx1 = np.clip(idx0, 0, len(gyro_quats)-1), np.clip(idx1, 0, len(gyro_quats)-1)
        q0 = quat_normalize(gyro_quats[idx0])
        q1 = quat_normalize(gyro_quats[idx1])
        q_rel = quat_multiply(quat_conjugate(q0), q1)
        raw_qrel.append([q_rel[1], q_rel[2], q_rel[3]])
    raw_qrel = np.array(raw_qrel)
    def safe_corr_diag(a, b):
        if len(a) < 5: return 0.0
        return np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0
    best_corr = -1
    best_map = ""
    names = ['qx', 'qy', 'qz']
    combos = [(gx, gy, sx, sy) for gx in range(3) for gy in range(3)
              if gx != gy for sx in [+1, -1] for sy in [+1, -1]]
    for gx_src, gy_src, gx_sign, gy_sign in combos:
        gx = raw_qrel[:n_diag, gx_src][diag_mask] * gx_sign
        gy = raw_qrel[:n_diag, gy_src][diag_mask] * gy_sign
        cx = safe_corr_diag(raw_disp[:n_diag, 0][diag_mask], gx)
        cy = safe_corr_diag(raw_disp[:n_diag, 1][diag_mask], gy)
        avg = (abs(cx) + abs(cy)) / 2
        label = f"dx={gx_sign:+d}*{names[gx_src]}, dy={gy_sign:+d}*{names[gy_src]}"
        if avg > 0.4:
            print(f"  {label}: corr_x={cx:.3f}, corr_y={cy:.3f}, avg={avg:.3f} ***")
        if avg > best_corr:
            best_corr = avg
            best_map = label
    print(f"  Best: {best_map} (avg={best_corr:.3f})")

    # Print summary
    summary = f"""
EIS Bench Test Results
======================
Raw jitter RMS:              {metrics['raw_jitter_rms_px']:.2f} px
Gyro raw RMS:                {metrics['gyro_raw_rms_px']:.2f} px
Scale ratio (optical/gyro):  {metrics['scale_ratio']:.2f}x
Causal stabilized residual:  {metrics['causal_residual_rms_px']:.2f} px
Ideal (non-causal) residual: {metrics['noncausal_residual_rms_px']:.2f} px
Improvement ratio (causal):  {metrics['improvement_ratio']:.2f}x
Improvement ratio (ideal):   {metrics['ideal_improvement_ratio']:.2f}x
Causal vs ideal gap:         {metrics['causal_vs_ideal_error_px']:.2f} px
Gyro-video correlation X:    {metrics['gyro_video_corr_x']:.3f}
Gyro-video correlation Y:    {metrics['gyro_video_corr_y']:.3f}
Frames analyzed:             {metrics['n_frames']} ({metrics.get('n_good_frames', 'N/A')} high-confidence)
FPS:                         {metrics['fps']:.1f}

INTERPRETATION:
  - Gyro-video correlation > 0.7 = good gyro-video alignment
  - Gyro-video correlation < 0.3 = timestamp mismatch, axis mapping error, or OIS interference
  - Scale ratio ~1.0 = focal length calibration is correct
  - Improvement ratio > 2x = stabilization is effective
  - Causal vs ideal gap = room for algorithm improvement
"""
    print(summary)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(summary)

    # Generate plots
    generate_plots(raw_disp, causal_pred, nc_pred, gyro_raw_disp, metrics,
                   out_dir, fps, width, height)

    # Generate stabilized video
    if args.output_video:
        print("\nGenerating stabilized video...")
        out_video = str(out_dir / "stabilized.mp4")
        generate_stabilized_video(args.raw_video, out_video, causal_h,
                                   gyro_ts, frame_ts)

    print("Done.")

if __name__ == "__main__":
    main()
