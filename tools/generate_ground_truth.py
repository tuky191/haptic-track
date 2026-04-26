#!/usr/bin/env python3
"""Generate ground-truth identity bboxes for a video used by VideoReplayTest.

Approach: YOLOv8 person detection + ByteTrack association across frames.
ByteTrack assigns persistent track IDs that survive brief occlusions and
fast motion — exactly the cases that stymie pixel-template trackers
(MIL, KCF) and naive frame-to-frame IoU matching.

Why independent ground truth: YOLOv8 + ByteTrack is a different model
family from the app's identity stack (EfficientDet + MobileNetV3 embedding
+ OSNet body re-id + ViT visual tracker). If the app's stack swaps subject
A for subject B and YOLO+ByteTrack stays on A, the test catches it via
VideoReplayTest.ReplayResult.wrongIdentityReacqs.

Limitations: ByteTrack itself can drift on long heavy occlusions or when
two similar subjects cross paths repeatedly. Sample preview frames in the
generated <video>_gt_preview/ directory and manually null-out post-drift
annotations in the .gt.json for any frames where the green GT box is on
the wrong person — the test loader skips frames where box is null.

Outputs:
  <video>.gt.json       per-frame normalized [l,t,r,b] for the lock subject,
                        or null when the track is absent (occluded / lost)
  <video>_gt_preview/   sampled preview frames with the GT box overlaid

Usage:
  cd tools && .venv/bin/python generate_ground_truth.py /path/to/video.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def iou(a, b):
    """IoU of two normalized [l,t,r,b] boxes."""
    al, at, ar, ab = a
    bl, bt, br, bb = b
    il, it = max(al, bl), max(at, bt)
    ir, ib = min(ar, br), min(ab, bb)
    if il >= ir or it >= ib:
        return 0.0
    inter = (ir - il) * (ib - it)
    a_area = (ar - al) * (ab - at)
    b_area = (br - bl) * (bb - bt)
    return inter / (a_area + b_area - inter)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to .mp4")
    parser.add_argument("--spec", help="Path to .json spec (default: <video>.json)")
    parser.add_argument("--out", help="Path to .gt.json output")
    parser.add_argument("--preview-every", type=int, default=30,
                        help="Save preview frames every N frames (default: 30)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 model file (default: yolov8n.pt at repo root)")
    parser.add_argument("--score", type=float, default=0.4,
                        help="Detection confidence floor (default: 0.4)")
    args = parser.parse_args()

    video_path = Path(args.video)
    spec_path = Path(args.spec) if args.spec else video_path.with_suffix(".json")
    out_path = Path(args.out) if args.out else \
        video_path.with_name(video_path.stem + ".gt.json")
    preview_dir = video_path.with_name(video_path.stem + "_gt_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model
    if not Path(model_path).is_absolute():
        # Try relative to script, then repo root
        candidates = [
            Path(__file__).parent / model_path,
            Path(__file__).parent.parent.parent / model_path,
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    spec = json.loads(spec_path.read_text())
    lock_frame = spec["lockFrame"]
    lock_box = spec["lockBox"]

    print(f"Loading YOLOv8: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}", file=sys.stderr)
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name} {w}x{h} frames={nframes}")
    print(f"Lock frame {lock_frame}: {lock_box}")
    cap.release()

    # First pass — track every frame, collect (frame, track_id, box) tuples
    # for class 0 (person in COCO). YOLO loads + tracks the whole video.
    print("Running YOLOv8 + ByteTrack on full video...")
    tracks_per_frame = {}  # frame_idx -> list of (track_id, box_n, score)
    for frame_idx, result in enumerate(
            model.track(source=str(video_path), classes=[0],
                        conf=args.score, persist=True,
                        stream=True, verbose=False)):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            tracks_per_frame[frame_idx] = []
            continue
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        items = []
        for tid, (x1, y1, x2, y2), s in zip(ids, xyxy, scores):
            items.append({
                "track_id": int(tid),
                "box": [x1 / w, y1 / h, x2 / w, y2 / h],
                "score": float(s),
            })
        tracks_per_frame[frame_idx] = items

    # Identify the lock track ID — the track at lock_frame whose box has
    # highest IoU with the spec's lockBox.
    lock_tracks = tracks_per_frame.get(lock_frame, [])
    if not lock_tracks:
        print(f"No tracks at lock frame {lock_frame}", file=sys.stderr)
        sys.exit(1)
    best_tid, best_iou = None, 0.0
    for t in lock_tracks:
        i = iou(t["box"], lock_box)
        if i > best_iou:
            best_tid, best_iou = t["track_id"], i
    print(f"Lock track id={best_tid} (IoU={best_iou:.3f} with spec lockBox)")
    if best_iou < 0.2:
        print(f"WARNING: best IoU is {best_iou:.3f} — spec lockBox doesn't "
              f"closely match any YOLO detection. Ground truth may be wrong.",
              file=sys.stderr)

    # Build annotations: for each frame, find the entry with track_id == best_tid
    annotations = []
    matched = lost = 0
    cap = cv2.VideoCapture(str(video_path))
    for frame_idx in range(nframes):
        ret, frame = cap.read()
        if not ret:
            break
        items = tracks_per_frame.get(frame_idx, [])
        gt = None
        for it in items:
            if it["track_id"] == best_tid:
                gt = it
                break
        if frame_idx < lock_frame:
            annotations.append({"frame": frame_idx, "box": None, "status": "pre-lock"})
        elif gt is None:
            annotations.append({"frame": frame_idx, "box": None, "status": "lost"})
            lost += 1
        else:
            annotations.append({
                "frame": frame_idx,
                "box": gt["box"],
                "status": "matched",
                "score": gt["score"],
            })
            matched += 1

        # Preview
        if frame_idx % args.preview_every == 0 and frame_idx >= lock_frame:
            preview = frame.copy()
            ann = annotations[-1]
            # Draw all detected persons
            for it in items:
                bx = [int(it["box"][0] * w), int(it["box"][1] * h),
                      int(it["box"][2] * w), int(it["box"][3] * h)]
                if it["track_id"] == best_tid:
                    color = (0, 255, 0)
                    width = 3
                else:
                    color = (100, 100, 255)
                    width = 1
                cv2.rectangle(preview, (bx[0], bx[1]), (bx[2], bx[3]), color, width)
                cv2.putText(preview, f"#{it['track_id']}",
                            (bx[0], max(15, bx[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            color_text = (0, 0, 255) if ann["box"] is None else (0, 255, 0)
            cv2.putText(preview, f"f{frame_idx} GT-id={best_tid} {ann['status']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)
            cv2.imwrite(str(preview_dir /
                            f"frame_{frame_idx:05d}_{ann['status']}.jpg"), preview)
    cap.release()

    out = {
        "video": video_path.name,
        "method": "YOLOv8 + ByteTrack",
        "lock_frame": lock_frame,
        "lock_box": lock_box,
        "lock_track_id": best_tid,
        "lock_track_iou": best_iou,
        "stats": {
            "matched": matched,
            "lost": lost,
            "total": len(annotations),
        },
        "annotations": annotations,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}: matched={matched}, lost={lost}, total={len(annotations)}")
    print(f"Preview frames: {preview_dir}/")


if __name__ == "__main__":
    main()
