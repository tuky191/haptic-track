#!/usr/bin/env python3
"""Extract track-tagged person crops from a video using YOLOv8 + ByteTrack.

Run YOLOv8 detection on every frame, propagate identities with ByteTrack, and
save each person bounding box as a PNG named ``frame_<N>_track_<T>.png``. The
output is the ground truth for a same-vs-different cosine benchmark — within
each track ID, all crops are the same person; across track IDs they are
different people (assuming ByteTrack didn't ID-switch, which is good enough
on the short well-lit clips we use).

Usage:
    cd tools && .venv/bin/python extract_person_crops.py \
        ../test_videos/kid_to_wife_panning.mp4 \
        --out crops/kid_to_wife --downscale 720

The `--downscale` flag resizes the long edge before running YOLO, matching
what the device sees. Default 720 is a reasonable proxy for the on-device
analysis frame size (~640).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path, help="Input video file")
    p.add_argument("--out", type=Path, required=True, help="Output crop directory")
    p.add_argument("--weights", type=Path,
                   default=Path(__file__).parent.parent / "yolov8n.pt",
                   help="YOLO weights (default: yolov8n.pt at repo root)")
    p.add_argument("--downscale", type=int, default=720,
                   help="Resize video long edge to this size before tracking (0=off)")
    p.add_argument("--min-side-px", type=int, default=64,
                   help="Skip crops with min(w,h) below this (filter low-res detections)")
    p.add_argument("--frame-stride", type=int, default=2,
                   help="Skip every Nth frame to reduce dataset size")
    p.add_argument("--max-per-track", type=int, default=80,
                   help="Cap crops per track ID (uniform sampling across the timeline)")
    p.add_argument("--conf", type=float, default=0.4)
    return p.parse_args()


def downscale_box(box_xyxy, scale):
    return [c / scale for c in box_xyxy]


def main():
    args = parse_args()
    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    # Lazy import — torch/ultralytics are slow to load.
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"failed to open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.downscale and max(src_w, src_h) > args.downscale:
        scale = args.downscale / max(src_w, src_h)
    else:
        scale = 1.0
    proc_w = int(src_w * scale)
    proc_h = int(src_h * scale)

    print(f"video: {args.video.name} src={src_w}x{src_h} proc={proc_w}x{proc_h} "
          f"frames={n_frames}")

    track_crops = defaultdict(list)  # track_id -> [(frame_idx, box_xyxy_native, crop_bgr)]
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % args.frame_stride != 0:
            continue

        proc = cv2.resize(frame, (proc_w, proc_h)) if scale != 1.0 else frame

        # Track persons (class 0 in COCO). ByteTrack is bundled with ultralytics.
        results = model.track(
            proc, classes=[0], conf=args.conf,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )
        if not results or len(results) == 0:
            continue
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        for box_proc, tid in zip(boxes, ids):
            # Map back to native coords for full-resolution crop.
            x1, y1, x2, y2 = (int(c / scale) for c in box_proc)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(src_w, x2); y2 = min(src_h, y2)
            w, h = x2 - x1, y2 - y1
            if min(w, h) < args.min_side_px:
                continue
            crop = frame[y1:y2, x1:x2]
            track_crops[tid].append((frame_idx, (x1, y1, x2, y2), crop))

    cap.release()

    # Cap per-track to args.max_per_track via uniform sampling.
    per_track_count = {}
    saved = 0
    for tid, items in sorted(track_crops.items()):
        if len(items) > args.max_per_track:
            stride = len(items) / args.max_per_track
            kept = [items[int(i * stride)] for i in range(args.max_per_track)]
        else:
            kept = items
        per_track_count[tid] = len(kept)
        for fidx, box, crop in kept:
            fname = args.out / f"frame_{fidx:05d}_track_{tid:03d}.png"
            cv2.imwrite(str(fname), crop)
            saved += 1

    # Manifest (JSON-friendly summary).
    print()
    print(f"saved {saved} crops across {len(per_track_count)} tracks → {args.out}")
    print(f"per-track counts (sorted by count desc):")
    for tid, n in sorted(per_track_count.items(), key=lambda kv: -kv[1]):
        print(f"  track {tid:>3d}: {n:>4d} crops")

    # Note tracks that ID-switched would inflate "different person" pairs across
    # the same individual; we sanity-check by looking at very high cosines later.


if __name__ == "__main__":
    main()
