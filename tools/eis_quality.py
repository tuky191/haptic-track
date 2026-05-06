#!/usr/bin/env python3
"""
Standalone video stabilization quality measurement.

Computes the Liu triplet (Stability Score, Cropping Ratio, Distortion) and
jitter metrics from one or more video files. No gyro data needed — works on
any video recording. Use this to compare raw, ISP-stabilized, and gyro EIS
videos side-by-side.

Based on: Liu et al., "Bundled Camera Paths for Video Stabilization",
SIGGRAPH 2013. Reference implementation: DIFRINT metrics.py.

Usage:
    python eis_quality.py video1.mp4 video2.mp4 ...
    python eis_quality.py --raw raw.mp4 --isp isp.mp4 --gyro gyro.mp4
    python eis_quality.py *.mp4 --output results/

Metrics:
    S (Stability Score)  — frequency-domain smoothness [0,1], higher = smoother
    C (Cropping Ratio)   — FOV preservation [0,1], higher = less crop
    D (Distortion)       — geometric fidelity [0,1], higher = less distortion
    Jitter RMS           — inter-frame displacement in pixels, lower = smoother
    Accumulated Flow     — total inter-frame motion in pixels, lower = smoother
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def stability_score_from_homographies(H_seq):
    """Liu Stability Score from a sequence of inter-frame homographies."""
    if len(H_seq) < 8:
        return 0.0, 0.0, 0.0

    P_seq = []
    Pt = np.eye(3)
    for H in H_seq:
        Pt = Pt @ H
        P_seq.append(Pt.copy())

    trans = np.array([np.sqrt(P[0, 2]**2 + P[1, 2]**2) for P in P_seq])
    rot = np.array([np.degrees(np.arctan2(P[1, 0], P[0, 0])) for P in P_seq])

    def freq_ratio(signal):
        fft = np.fft.fft(signal)
        power = np.abs(fft)**2
        power = power[1:len(power)//2]
        if len(power) < 6 or np.sum(power) < 1e-12:
            return 1.0
        return float(np.sum(power[:5]) / np.sum(power))

    ss_t = freq_ratio(trans)
    ss_r = freq_ratio(rot)
    return (ss_t + ss_r) / 2, ss_t, ss_r


def analyze_video(video_path: str):
    """Compute all quality metrics from a single video file.

    Returns dict with stability_score, stability_t, stability_r,
    cropping_ratio, distortion, jitter_rms_px, accumulated_flow,
    n_frames, fps, width, height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}", file=sys.stderr)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    MIN_MATCH = 10
    RATIO = 0.7
    RANSAC_THRESH = 5.0

    prev_gray = None
    prev_kp = None
    prev_desc = None

    inter_frame_H = []
    jitter_magnitudes = []
    pc_magnitudes = []
    cropping_ratios = []
    distortion_values = []
    accumulated_flow = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        frame_count += 1

        if prev_gray is not None:
            # Phase correlation (robust global translation estimate)
            (dx, dy), resp = cv2.phaseCorrelate(
                prev_gray.astype(np.float64), gray.astype(np.float64))
            if resp > 0.10:
                pc_magnitudes.append(np.sqrt(dx**2 + dy**2))

            if desc is not None and prev_desc is not None:
                matches = bf.knnMatch(prev_desc, desc, k=2)
                good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance]

                if len(good) >= MIN_MATCH:
                    src = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
                    if H is not None:
                        inter_frame_H.append(H)

                        tx, ty = H[0, 2], H[1, 2]
                        mag = np.sqrt(tx**2 + ty**2)
                        jitter_magnitudes.append(mag)
                        accumulated_flow += mag

                        scale = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
                        cr = min(1.0 / scale if scale > 0 else 1.0, 1.0)
                        cropping_ratios.append(cr)

                        affine = H[0:2, 0:2]
                        try:
                            w, _ = np.linalg.eig(affine)
                            w_abs = np.sort(np.abs(w))[::-1]
                            if w_abs[0] > 1e-10:
                                dv = float(np.real(w_abs[1] / w_abs[0]))
                                dv = max(0.0, min(1.0, abs(dv)))
                                distortion_values.append(dv)
                        except np.linalg.LinAlgError:
                            pass

        if frame_count % 100 == 0:
            print(f"  {frame_count}/{total} frames...", end='\r')

        prev_gray = gray
        prev_kp = kp
        prev_desc = desc

    cap.release()
    print(f"  {frame_count} frames processed    ")

    if len(inter_frame_H) < 8:
        print(f"  Too few matched frames ({len(inter_frame_H)}) for metrics")
        return None

    s_avg, s_t, s_r = stability_score_from_homographies(np.array(inter_frame_H))
    jitter_arr = np.array(jitter_magnitudes)
    jitter_rms = float(np.sqrt(np.mean(jitter_arr**2)))
    pc_arr = np.array(pc_magnitudes) if pc_magnitudes else np.array([0.0])
    pc_rms = float(np.sqrt(np.mean(pc_arr**2)))

    return {
        "stability_score": s_avg,
        "stability_t": s_t,
        "stability_r": s_r,
        "cropping_ratio_mean": float(np.mean(cropping_ratios)) if cropping_ratios else 1.0,
        "cropping_ratio_min": float(np.min(cropping_ratios)) if cropping_ratios else 1.0,
        "distortion": float(np.min(distortion_values)) if distortion_values else 1.0,
        "jitter_rms_px": jitter_rms,
        "jitter_p95_px": float(np.percentile(jitter_arr, 95)),
        "jitter_max_px": float(np.max(jitter_arr)),
        "pc_jitter_rms_px": pc_rms,
        "accumulated_flow": accumulated_flow,
        "n_frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure video stabilization quality (Liu SIGGRAPH 2013)")
    parser.add_argument("videos", nargs="*", help="Video files to analyze")
    parser.add_argument("--raw", help="Raw (unstabilized) video — labeled as baseline")
    parser.add_argument("--isp", help="ISP-stabilized video — labeled as target")
    parser.add_argument("--gyro", help="Gyro EIS video — labeled as test")
    parser.add_argument("--output", help="Output directory for results (default: print to stdout)")
    args = parser.parse_args()

    # Build list of (label, path) pairs
    videos = []
    if args.raw:
        videos.append(("Raw", args.raw))
    if args.isp:
        videos.append(("ISP (target)", args.isp))
    if args.gyro:
        videos.append(("Gyro EIS", args.gyro))
    for v in (args.videos or []):
        label = Path(v).stem
        if not any(p == v for _, p in videos):
            videos.append((label, v))

    if not videos:
        parser.print_help()
        sys.exit(1)

    results = []
    for label, path in videos:
        print(f"\nAnalyzing: {label} ({path})")
        r = analyze_video(path)
        if r:
            r["label"] = label
            r["path"] = path
            results.append(r)

    if not results:
        print("\nNo videos could be analyzed.")
        sys.exit(1)

    # Print comparison table
    lines = []
    lines.append("")
    lines.append("Video Stabilization Quality Report")
    lines.append("=" * 80)
    lines.append("Metrics: S=Stability Score (higher=smoother), C=Cropping Ratio,")
    lines.append("         D=Distortion (higher=less warping), Jitter=inter-frame RMS")
    lines.append("")

    hdr = f"  {'Video':<20s}  {'S':>6s}  {'PC jitter':>10s}  {'SIFT jitter':>12s}  {'P95':>8s}  {'Flow':>9s}  {'Frames':>6s}"
    lines.append(hdr)
    lines.append(f"  {'-'*20}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*6}")

    for r in results:
        tag = r['label']
        lines.append(
            f"  {tag:<20s}  "
            f"{r['stability_score']:6.4f}  "
            f"{r['pc_jitter_rms_px']:7.2f} px  "
            f"{r['jitter_rms_px']:9.2f} px  "
            f"{r['jitter_p95_px']:5.2f} px  "
            f"{r['accumulated_flow']:6.1f} px  "
            f"{r['n_frames']:6d}"
        )

    # Improvement ratios if we have a raw baseline
    raw_r = next((r for r in results if r['label'] == 'Raw'), None)
    if raw_r and len(results) > 1:
        lines.append("")
        lines.append("Improvement vs Raw:")
        for r in results:
            if r['label'] == 'Raw':
                continue
            s_imp = r['stability_score'] / raw_r['stability_score'] if raw_r['stability_score'] > 0 else float('inf')
            pc_imp = raw_r['pc_jitter_rms_px'] / r['pc_jitter_rms_px'] if r['pc_jitter_rms_px'] > 0 else float('inf')
            j_imp = raw_r['jitter_rms_px'] / r['jitter_rms_px'] if r['jitter_rms_px'] > 0 else float('inf')
            f_imp = raw_r['accumulated_flow'] / r['accumulated_flow'] if r['accumulated_flow'] > 0 else float('inf')
            lines.append(f"  {r['label']:<20s}  S: {s_imp:.2f}x  PC: {pc_imp:.2f}x  SIFT: {j_imp:.2f}x  Flow: {f_imp:.2f}x")

    lines.append("")
    summary = "\n".join(lines)
    print(summary)

    # Save to file
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "quality_report.txt", "w") as f:
            f.write(summary)
        print(f"Report saved to {out_dir / 'quality_report.txt'}")

        # Generate comparison bar chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            labels_plot = [r['label'] for r in results]
            colors = ['#cc4444' if 'Raw' in l else '#44aa44' if 'ISP' in l
                      else '#4488cc' for l in labels_plot]

            # Stability Score (higher = better)
            vals = [r['stability_score'] for r in results]
            bars = axes[0].bar(labels_plot, vals, color=colors, alpha=0.8)
            for b, v in zip(bars, vals):
                axes[0].text(b.get_x() + b.get_width()/2, b.get_height(),
                           f"{v:.4f}", ha='center', va='bottom', fontsize=9)
            axes[0].set_ylabel("Stability Score (S)")
            axes[0].set_title("Stability (higher = smoother)")

            # Jitter RMS (lower = better)
            vals = [r['jitter_rms_px'] for r in results]
            bars = axes[1].bar(labels_plot, vals, color=colors, alpha=0.8)
            for b, v in zip(bars, vals):
                axes[1].text(b.get_x() + b.get_width()/2, b.get_height(),
                           f"{v:.1f}", ha='center', va='bottom', fontsize=9)
            axes[1].set_ylabel("Jitter RMS (px)")
            axes[1].set_title("Jitter (lower = smoother)")

            # Accumulated Flow (lower = better)
            vals = [r['accumulated_flow'] for r in results]
            bars = axes[2].bar(labels_plot, vals, color=colors, alpha=0.8)
            for b, v in zip(bars, vals):
                axes[2].text(b.get_x() + b.get_width()/2, b.get_height(),
                           f"{v:.0f}", ha='center', va='bottom', fontsize=9)
            axes[2].set_ylabel("Accumulated Flow (px)")
            axes[2].set_title("Total Motion (lower = smoother)")

            plt.suptitle("Video Stabilization Quality Comparison", fontsize=14)
            plt.tight_layout()
            plt.savefig(out_dir / "quality_comparison.png", dpi=150)
            plt.close()
            print(f"Chart saved to {out_dir / 'quality_comparison.png'}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
