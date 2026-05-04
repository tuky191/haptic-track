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
    """Affine stabilization matrix: crop zoom + center translation. Returns 3×3 row-major.

    Computes the full homography H = K × R × K⁻¹ to find the center displacement,
    then builds an affine matrix with constant scale (no perspective/breathing)."""
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
    # Center displacement with perspective division
    w  = h[2,0]*0.5 + h[2,1]*0.5 + h[2,2]
    cu = (h[0,0]*0.5 + h[0,1]*0.5 + h[0,2]) / w
    cv = (h[1,0]*0.5 + h[1,1]*0.5 + h[1,2]) / w
    du = cu - 0.5
    dv = cv - 0.5

    # Affine: constant crop zoom + translation (no perspective/scale variation)
    iz = 1.0 / zoom
    tx = 0.5 * (1.0 - iz) + iz * du
    ty = 0.5 * (1.0 - iz) + iz * dv
    return np.array([
        [iz,  0.0, tx],
        [0.0, iz,  ty],
        [0.0, 0.0, 1.0],
    ])

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
    ois_compensation: float = 1.0

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
        ois_compensation=float(row.get("oisCompensation", "1.0")),
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

def load_corrections(bench_dir: Path):
    """Load per-frame device corrections from corrections.csv if present.
    Returns dict with arrays, or None if file doesn't exist."""
    corr_file = bench_dir / "corrections.csv"
    if not corr_file.exists():
        return None
    with open(corr_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    n = len(rows)
    result = {
        'frame_idx': np.zeros(n, dtype=np.int64),
        'timestamp_ns': np.zeros(n, dtype=np.int64),
        'raw_quat': np.zeros((n, 4)),
        'smooth_quat': np.zeros((n, 4)),
        'eff_tc': np.zeros(n),
        'corr_deg': np.zeros(n),
        'leash': np.zeros(n, dtype=bool),
        'matrix': np.zeros((n, 9)),
    }
    for i, row in enumerate(rows):
        result['frame_idx'][i] = int(row['frame_idx'])
        result['timestamp_ns'][i] = int(row['timestamp_ns'])
        result['raw_quat'][i] = [float(row['raw_w']), float(row['raw_x']),
                                  float(row['raw_y']), float(row['raw_z'])]
        result['smooth_quat'][i] = [float(row['smooth_w']), float(row['smooth_x']),
                                     float(row['smooth_y']), float(row['smooth_z'])]
        result['eff_tc'][i] = float(row['eff_tc'])
        result['corr_deg'][i] = float(row['corr_deg'])
        result['leash'][i] = int(row['leash']) != 0
        result['matrix'][i] = [float(row[f'm{j}']) for j in range(9)]
    return result

# ---------------------------------------------------------------------------
# Adaptive smoothing constants (must match GyroStabilizer.kt)
# ---------------------------------------------------------------------------

PAN_VELOCITY_THRESHOLD_DEG = 15.0
PAN_ONSET_SEC = 0.20
PAN_TC_FACTOR = 0.30
VELOCITY_SMOOTHING_TC = 0.05
TC_RAMP_SPEED = 5.0

# ---------------------------------------------------------------------------
# Algorithm replay  — exact match to GyroStabilizer.onSensorChanged
# ---------------------------------------------------------------------------

def replay_stabilization(gyro_ts, gyro_quats, params: BenchParams, sensor_orientation=90,
                         adaptive=True):
    """
    Replays the stabilization algorithm on gyro data.
    Returns (corrections_h, smoothed_quats, adaptive_trace).
    adaptive_trace is a dict with velocity/effective_tc/is_panning arrays, or None.
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

    # Adaptive smoothing state
    prev_raw = None
    smoothed_vel_deg = 0.0
    high_vel_duration = 0.0
    effective_tc = params.time_constant

    trace_vel = np.zeros(n)
    trace_etc = np.full(n, params.time_constant)
    trace_pan = np.zeros(n, dtype=bool)

    for i in range(n):
        raw = quat_normalize(gyro_quats[i])
        now_ns = gyro_ts[i]

        if not initialized:
            smoothed = raw.copy()
            initialized = True
            last_ts = now_ns
            prev_raw = raw.copy()
            smoothed_vel_deg = 0.0
            high_vel_duration = 0.0
            effective_tc = params.time_constant
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
            prev_raw = raw.copy()
            smoothed_vel_deg = 0.0
            high_vel_duration = 0.0
            effective_tc = params.time_constant
            smoothed_quats[i] = smoothed
            corrections_h[i] = identity
            continue

        dt_sec = dt_ns / 1e9
        sample_rate = 0.95 * sample_rate + 0.05 * (1.0 / dt_sec)

        if adaptive and prev_raw is not None:
            delta_quat = quat_multiply(quat_conjugate(prev_raw), raw)
            delta_angle = 2.0 * np.arccos(np.clip(abs(delta_quat[0]), 0.0, 1.0))
            ang_vel_deg = np.degrees(delta_angle) / dt_sec

            vel_alpha = 1.0 - np.exp(-dt_sec / VELOCITY_SMOOTHING_TC)
            smoothed_vel_deg += vel_alpha * (ang_vel_deg - smoothed_vel_deg)

            if smoothed_vel_deg > PAN_VELOCITY_THRESHOLD_DEG:
                high_vel_duration += dt_sec
            else:
                high_vel_duration = 0.0

            is_panning = high_vel_duration >= PAN_ONSET_SEC
            target_tc = params.time_constant * PAN_TC_FACTOR if is_panning else params.time_constant
            tc_alpha = 1.0 - np.exp(-dt_sec * TC_RAMP_SPEED)
            effective_tc += tc_alpha * (target_tc - effective_tc)

            trace_vel[i] = smoothed_vel_deg
            trace_etc[i] = effective_tc
            trace_pan[i] = is_panning
        else:
            effective_tc = params.time_constant

        prev_raw = raw.copy()

        alpha = 1.0 - np.exp(-(1.0 / sample_rate) / effective_tc)
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
        if params.ois_compensation < 1.0:
            identity_q = np.array([1.0, 0.0, 0.0, 0.0])
            correction_device = slerp(identity_q, correction_device, params.ois_compensation)
        correction = quat_multiply(quat_multiply(d2s_quat, correction_device),
                                   quat_conjugate(d2s_quat))
        R = quat_to_rotation_matrix(correction)
        h = compute_homography_uv(R, params.fx_uv, params.fy_uv, params.crop_zoom)

        corrections_h[i] = h

    adaptive_trace = {
        'velocity_deg': trace_vel,
        'effective_tc': trace_etc,
        'is_panning': trace_pan,
    } if adaptive else None

    return corrections_h, smoothed_quats, adaptive_trace

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


def generate_adaptive_plot(gyro_ts, adaptive_trace, out_dir: Path):
    """Plot adaptive smoothing diagnostics: velocity, effective TC, pan segments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping adaptive plot")
        return

    t = (gyro_ts - gyro_ts[0]) / 1e9
    vel = adaptive_trace['velocity_deg']
    etc = adaptive_trace['effective_tc']
    pan = adaptive_trace['is_panning']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(t, vel, alpha=0.7, linewidth=0.5)
    ax1.axhline(y=PAN_VELOCITY_THRESHOLD_DEG, color='r', linestyle='--', alpha=0.5,
                label=f'Threshold ({PAN_VELOCITY_THRESHOLD_DEG}°/s)')
    pan_start = None
    for i in range(len(pan)):
        if pan[i] and pan_start is None:
            pan_start = t[i]
        elif not pan[i] and pan_start is not None:
            ax1.axvspan(pan_start, t[i], alpha=0.15, color='orange')
            pan_start = None
    if pan_start is not None:
        ax1.axvspan(pan_start, t[-1], alpha=0.15, color='orange')
    ax1.set_ylabel('Angular velocity (°/s)')
    ax1.set_title('Adaptive smoothing: velocity and effective TC')
    ax1.legend()

    ax2.plot(t, etc, alpha=0.7, linewidth=0.5, color='green', label='Effective TC')
    ax2.axhline(y=etc[0], color='gray', linestyle=':', alpha=0.5, label=f'Base TC ({etc[0]:.3f})')
    ax2.set_ylabel('Effective TC (s)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()

    pan_pct = 100.0 * pan.sum() / max(len(pan), 1)
    fig.suptitle(f'Pan detection: {pan.sum()} samples ({pan_pct:.1f}%), '
                 f'peak vel: {vel.max():.1f}°/s', y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / 'adaptive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Adaptive diagnostic plot saved to {out_dir / 'adaptive.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sensor_to_portrait_gl_colmajor(H_sensor_rowmajor, orientation):
    """Convert sensor-UV row-major homography to portrait GL column-major mat3.
    Exact port of GyroStabilizer.sensorToPortraitGL() + row-to-col conversion."""
    # First convert row-major to GL column-major (same space)
    h = H_sensor_rowmajor
    h00, h01, h02 = h[0,0], h[0,1], h[0,2]
    h10, h11, h12 = h[1,0], h[1,1], h[1,2]
    h20, h21, h22 = h[2,0], h[2,1], h[2,2]

    if orientation not in (90, 270):
        return np.array([h00, h10, h20, h01, h11, h21, h02, h12, h22], dtype=np.float32)

    if orientation == 90:
        p00 = h11;        p01 = -h10;       p02 = h10 + h12
        p10 = h21 - h01;  p11 = h00 - h20;  p12 = h20 + h22 - h00 - h02
        p20 = h21;        p21 = -h20;        p22 = h20 + h22
    else:
        p00 = h11 - h21;  p01 = h20 - h10;  p02 = h21 + h22 - h11 - h12
        p10 = -h01;       p11 = h00;         p12 = h01 + h02
        p20 = -h21;       p21 = h20;         p22 = h21 + h22

    return np.array([p00, p10, p20, p01, p11, p21, p02, p12, p22], dtype=np.float32)


def gl_colmajor_center_disp(m):
    """Extract center-pixel displacement from GL column-major mat3."""
    tu = m[0]*0.5 + m[3]*0.5 + m[6]
    tv = m[1]*0.5 + m[4]*0.5 + m[7]
    return tu - 0.5, tv - 0.5


def compare_device_vs_replay(corrections_data, corrections_h, gyro_ts, frame_ts,
                              sensor_orientation, out_dir):
    """Compare device-logged corrections against replay-computed ones.
    Prints diagnostics and generates a comparison plot."""
    dev = corrections_data
    n_dev = len(dev['frame_idx'])
    n_replay = len(frame_ts)
    n = min(n_dev, n_replay)
    if n == 0:
        print("  No frames to compare")
        return

    dev_disp_u = np.zeros(n)
    dev_disp_v = np.zeros(n)
    rep_disp_u = np.zeros(n)
    rep_disp_v = np.zeros(n)
    quat_angle_diff = np.zeros(n)
    smooth_angle_diff = np.zeros(n)

    for i in range(n):
        # Device center displacement
        dev_du, dev_dv = gl_colmajor_center_disp(dev['matrix'][i])
        dev_disp_u[i] = dev_du
        dev_disp_v[i] = dev_dv

        # Replay: find gyro correction at frame time, convert to portrait GL
        ts = dev['timestamp_ns'][i]
        idx = np.searchsorted(gyro_ts, ts, side='right') - 1
        idx = np.clip(idx, 0, len(corrections_h) - 1)
        rep_mat = sensor_to_portrait_gl_colmajor(corrections_h[idx], sensor_orientation)
        rep_du, rep_dv = gl_colmajor_center_disp(rep_mat)
        rep_disp_u[i] = rep_du
        rep_disp_v[i] = rep_dv

        # Quaternion comparison
        dev_raw = quat_normalize(dev['raw_quat'][i])
        rep_raw = quat_normalize(gyro_ts)  # need the actual quats
        dev_smooth = quat_normalize(dev['smooth_quat'][i])

    # Summary
    err_u = dev_disp_u - rep_disp_u
    err_v = dev_disp_v - rep_disp_v
    err_mag = np.sqrt(err_u**2 + err_v**2)
    dev_mag = np.sqrt(dev_disp_u**2 + dev_disp_v**2)
    rep_mag = np.sqrt(rep_disp_u**2 + rep_disp_v**2)

    print(f"\nDevice vs Replay correction comparison ({n} frames):")
    print(f"  Device correction RMS:  {np.sqrt(np.mean(dev_mag**2)):.6f} UV")
    print(f"  Replay correction RMS:  {np.sqrt(np.mean(rep_mag**2)):.6f} UV")
    print(f"  Difference RMS:         {np.sqrt(np.mean(err_mag**2)):.6f} UV")
    print(f"  Difference max:         {err_mag.max():.6f} UV")

    if np.sqrt(np.mean(rep_mag**2)) > 1e-6:
        corr_u = np.corrcoef(dev_disp_u, rep_disp_u)[0,1] if np.std(dev_disp_u) > 0 else 0
        corr_v = np.corrcoef(dev_disp_v, rep_disp_v)[0,1] if np.std(dev_disp_v) > 0 else 0
        print(f"  Correlation:            U={corr_u:.4f}  V={corr_v:.4f}")

        scale_u = np.std(dev_disp_u) / np.std(rep_disp_u) if np.std(rep_disp_u) > 0 else 0
        scale_v = np.std(dev_disp_v) / np.std(rep_disp_v) if np.std(rep_disp_v) > 0 else 0
        print(f"  Scale (dev/replay):     U={scale_u:.4f}  V={scale_v:.4f}")

    print(f"  Device leash active:    {dev['leash'].sum()}/{n} frames ({100.0*dev['leash'].sum()/n:.1f}%)")
    print(f"  Device effective TC:    mean={dev['eff_tc'].mean():.4f}  min={dev['eff_tc'].min():.4f}  max={dev['eff_tc'].max():.4f}")
    print(f"  Device correction deg:  mean={dev['corr_deg'].mean():.3f}  max={dev['corr_deg'].max():.3f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        t = np.arange(n) / 30.0  # approximate time

        axes[0].plot(t, dev_disp_u, alpha=0.7, label='Device U', linewidth=0.8)
        axes[0].plot(t, rep_disp_u, alpha=0.7, label='Replay U', linewidth=0.8, linestyle='--')
        axes[0].set_ylabel('Center displacement U')
        axes[0].legend()
        axes[0].set_title('Device vs Replay: center pixel correction (portrait UV)')

        axes[1].plot(t, dev_disp_v, alpha=0.7, label='Device V', linewidth=0.8)
        axes[1].plot(t, rep_disp_v, alpha=0.7, label='Replay V', linewidth=0.8, linestyle='--')
        axes[1].set_ylabel('Center displacement V')
        axes[1].legend()

        axes[2].plot(t, err_mag, alpha=0.7, color='red', linewidth=0.8, label='|error|')
        axes[2].fill_between(t, 0, err_mag, alpha=0.2, color='red')
        leash_mask = dev['leash'][:n]
        if leash_mask.any():
            for start_i in range(n):
                if leash_mask[start_i]:
                    axes[2].axvline(t[start_i], alpha=0.1, color='orange', linewidth=0.5)
        axes[2].set_ylabel('|Device - Replay| UV')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(out_dir / 'device_vs_replay.png', dpi=150)
        plt.close()
        print(f"  Device vs replay plot saved to {out_dir / 'device_vs_replay.png'}")
    except ImportError:
        pass


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
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive smoothing (fixed TC only)")
    parser.add_argument("--isp-video", type=str, default=None,
                        help="ISP-stabilized reference video (quality target)")
    parser.add_argument("--gyro-video", type=str, default=None,
                        help="On-device gyro EIS video (measures current device quality)")
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

    # Measure reference videos (ISP and/or on-device gyro)
    isp_rms = None
    gyro_dev_rms = None
    isp_disp = None
    gyro_dev_disp = None
    if args.isp_video:
        print("\nMeasuring ISP reference video displacement...")
        isp_disp_raw, isp_fps, isp_w, isp_h = measure_frame_displacements(args.isp_video)
        isp_mag = np.sqrt(isp_disp_raw[:, 0]**2 + isp_disp_raw[:, 1]**2)
        isp_rms = np.sqrt(np.mean(isp_mag**2))
        # Normalize to raw video scale if resolutions differ
        isp_scale = (width / isp_w + height / isp_h) / 2.0
        isp_rms_scaled = isp_rms * isp_scale
        isp_disp = isp_disp_raw
        print(f"  ISP jitter RMS: {isp_rms:.2f} px ({isp_w}x{isp_h})"
              + (f" → {isp_rms_scaled:.2f} px scaled to {width}x{height}" if abs(isp_scale - 1.0) > 0.05 else ""))
        if abs(isp_scale - 1.0) > 0.05:
            isp_rms = isp_rms_scaled

    if args.gyro_video:
        print("\nMeasuring on-device gyro EIS video displacement...")
        gyro_dev_disp_raw, gd_fps, gd_w, gd_h = measure_frame_displacements(args.gyro_video)
        gd_mag = np.sqrt(gyro_dev_disp_raw[:, 0]**2 + gyro_dev_disp_raw[:, 1]**2)
        gyro_dev_rms = np.sqrt(np.mean(gd_mag**2))
        gd_scale = (width / gd_w + height / gd_h) / 2.0
        gyro_dev_disp = gyro_dev_disp_raw
        print(f"  Gyro device jitter RMS: {gyro_dev_rms:.2f} px ({gd_w}x{gd_h})"
              + (f" → {gyro_dev_rms * gd_scale:.2f} px scaled to {width}x{height}" if abs(gd_scale - 1.0) > 0.05 else ""))
        if abs(gd_scale - 1.0) > 0.05:
            gyro_dev_rms = gyro_dev_rms * gd_scale

    # Compute raw gyro inter-frame displacement (for correlation check)
    print("\nComputing raw gyro displacements...")
    gyro_raw_disp = raw_gyro_frame_displacements(
        gyro_ts, gyro_quats, frame_ts, params, args.sensor_orientation)
    print(f"  {len(gyro_raw_disp)} inter-frame gyro displacements")

    # Replay stabilization — fixed TC baseline
    print("Replaying causal stabilization (fixed TC)...")
    fixed_h, fixed_smoothed, _ = replay_stabilization(
        gyro_ts, gyro_quats, params, args.sensor_orientation, adaptive=False)
    fixed_pred = corrections_to_frame_displacements(fixed_h, gyro_ts, frame_ts, width, height)
    print(f"  {len(fixed_pred)} inter-frame predictions")

    # Replay stabilization — adaptive TC
    use_adaptive = not args.no_adaptive
    adaptive_trace = None
    if use_adaptive:
        print("Replaying causal stabilization (adaptive TC)...")
        adaptive_h, adaptive_smoothed, adaptive_trace = replay_stabilization(
            gyro_ts, gyro_quats, params, args.sensor_orientation, adaptive=True)
        adaptive_pred = corrections_to_frame_displacements(adaptive_h, gyro_ts, frame_ts, width, height)
        pan_pct = 100.0 * adaptive_trace['is_panning'].sum() / max(len(adaptive_trace['is_panning']), 1)
        print(f"  Pan detected: {pan_pct:.1f}% of samples, peak vel: {adaptive_trace['velocity_deg'].max():.1f}°/s")

    causal_h = adaptive_h if use_adaptive else fixed_h
    causal_pred = adaptive_pred if use_adaptive else fixed_pred

    # Replay non-causal ideal
    print("Computing non-causal ideal...")
    nc_h, nc_smoothed = replay_noncausal(
        gyro_ts, gyro_quats, params, args.sensor_orientation)
    nc_pred = corrections_to_frame_displacements(nc_h, gyro_ts, frame_ts, width, height)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(raw_disp, causal_pred, nc_pred, gyro_raw_disp, fps, width, height)
    metrics_fixed = compute_metrics(raw_disp, fixed_pred, nc_pred, gyro_raw_disp, fps, width, height)

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
    raw_rms = metrics['raw_jitter_rms_px']
    fixed_rms = metrics_fixed['causal_residual_rms_px']
    fixed_imp = metrics_fixed['improvement_ratio']
    adaptive_rms = metrics['causal_residual_rms_px'] if use_adaptive else fixed_rms
    adaptive_imp = metrics['improvement_ratio'] if use_adaptive else fixed_imp
    nc_rms = metrics['noncausal_residual_rms_px']
    nc_imp = metrics['ideal_improvement_ratio']

    lines = []
    lines.append("")
    lines.append("EIS Bench Test Results")
    lines.append("=" * 60)
    lines.append("")

    # Reference videos section
    if isp_rms is not None or gyro_dev_rms is not None:
        lines.append("Reference videos (separate recordings):")
        if isp_rms is not None:
            lines.append(f"  ISP stabilized:          {isp_rms:6.2f} px  ← TARGET")
        if gyro_dev_rms is not None:
            lines.append(f"  Gyro EIS (on-device):     {gyro_dev_rms:6.2f} px")
        lines.append("")

    # Main analysis table
    lines.append(f"Raw video analysis (scale ratio: {metrics['scale_ratio']:.2f}x):")
    lines.append(f"  Raw jitter:              {raw_rms:6.2f} px")
    lines.append(f"  Gyro replay (fixed TC):  {fixed_rms:6.2f} px  {fixed_imp:.2f}x improvement")
    if use_adaptive:
        adaptive_delta = (adaptive_imp / fixed_imp - 1.0) * 100 if fixed_imp > 0 else 0
        lines.append(f"  Gyro replay (adaptive):  {adaptive_rms:6.2f} px  {adaptive_imp:.2f}x improvement  ({adaptive_delta:+.1f}%)")
    lines.append(f"  Non-causal ideal:        {nc_rms:6.2f} px  {nc_imp:.2f}x improvement")

    # Gap analysis
    if isp_rms is not None:
        best_replay_rms = adaptive_rms if use_adaptive else fixed_rms
        gap = best_replay_rms - isp_rms
        gap_ratio = best_replay_rms / isp_rms if isp_rms > 0 else float('inf')
        lines.append("")
        lines.append(f"Gap to ISP target:         {gap:+.2f} px ({gap_ratio:.2f}x the ISP residual)")
        if nc_rms < isp_rms:
            lines.append(f"  Non-causal beats ISP — algorithm can theoretically reach target")
        else:
            lines.append(f"  Non-causal also above ISP — fundamental algorithm limit")

    lines.append("")
    lines.append(f"Gyro-video correlation:    X={metrics['gyro_video_corr_x']:.3f}  Y={metrics['gyro_video_corr_y']:.3f}")
    lines.append(f"Frames analyzed:           {metrics['n_frames']} ({metrics.get('n_good_frames', 'N/A')} high-confidence)")
    lines.append(f"FPS:                       {metrics['fps']:.1f}")
    lines.append("")

    summary = "\n".join(lines)
    print(summary)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(summary)

    # Generate plots
    generate_plots(raw_disp, causal_pred, nc_pred, gyro_raw_disp, metrics,
                   out_dir, fps, width, height)

    if adaptive_trace is not None:
        generate_adaptive_plot(gyro_ts, adaptive_trace, out_dir)

    # Comparison bar chart when reference videos are provided
    if isp_rms is not None or gyro_dev_rms is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = ["Raw"]
            values = [raw_rms]
            colors = ["#cc4444"]

            if isp_rms is not None:
                labels.append("ISP\n(target)")
                values.append(isp_rms)
                colors.append("#44aa44")

            if gyro_dev_rms is not None:
                labels.append("Gyro\n(device)")
                values.append(gyro_dev_rms)
                colors.append("#aa8844")

            labels.append("Gyro replay\n(fixed TC)")
            values.append(fixed_rms)
            colors.append("#4488cc")

            if use_adaptive:
                labels.append("Gyro replay\n(adaptive)")
                values.append(adaptive_rms)
                colors.append("#4466aa")

            labels.append("Non-causal\n(ideal)")
            values.append(nc_rms)
            colors.append("#888888")

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='white')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha='center', va='bottom', fontsize=10)
            if isp_rms is not None:
                ax.axhline(y=isp_rms, color='#44aa44', linestyle='--', alpha=0.5, label='ISP target')
                ax.legend()
            ax.set_ylabel("Jitter RMS (px)")
            ax.set_title("Stabilization quality comparison")
            plt.tight_layout()
            plt.savefig(out_dir / "comparison.png", dpi=150)
            plt.close()
            print(f"Comparison chart saved to {out_dir / 'comparison.png'}")
        except ImportError:
            pass

    # Compare device vs replay corrections if corrections.csv exists
    corrections_data = load_corrections(bench_dir)
    if corrections_data is not None:
        print(f"\nLoaded {len(corrections_data['frame_idx'])} device correction snapshots")
        # Use adaptive replay if device was running adaptive (check if TC varies)
        dev_tc_var = corrections_data['eff_tc'].std()
        if dev_tc_var > 0.001 and use_adaptive:
            print(f"  Device TC varies (std={dev_tc_var:.4f}) — comparing against adaptive replay")
            compare_h = adaptive_h
        else:
            print(f"  Device TC constant — comparing against fixed TC replay")
            compare_h = fixed_h
        compare_device_vs_replay(corrections_data, compare_h, gyro_ts, frame_ts,
                                  args.sensor_orientation, out_dir)
    else:
        print("\nNo corrections.csv found — capture a new session with the updated build for device-vs-replay comparison")

    # Generate stabilized video
    if args.output_video:
        print("\nGenerating stabilized video...")
        out_video = str(out_dir / "stabilized.mp4")
        generate_stabilized_video(args.raw_video, out_video, causal_h,
                                   gyro_ts, frame_ts)

    print("Done.")

if __name__ == "__main__":
    main()
