#!/usr/bin/env python3
"""Same-vs-different person face cosine benchmark.

Compares MobileFaceNet (current) vs EdgeFace-XS (candidate, IJCB 2023 winner)
on track-tagged person crops. Uses MediaPipe FaceDetection (BlazeFace, same as
on-device) to localize the face inside each person crop, applies a 5-keypoint
similarity transform alignment, and feeds both models 112×112 aligned faces.

Crops without a detected face are skipped — face is missing in many real
person detections (profile views, occlusions, distant subjects), and that's
data both models pay equally.

Usage:
    cd tools && ../.venv/bin/python benchmark_face_models.py \
        --crops-root crops --identity-map crops/identity_map.json \
        --video kid_to_wife
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Standard arc-face / insightface 5-point template at 112×112.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def percentile(xs, p):
    if not xs:
        return float("nan")
    return float(np.percentile(xs, p))


def parse_filename(name: str):
    parts = Path(name).stem.split("_")
    return int(parts[1]), int(parts[3])


# --------------------------------------------------------------------- face detection


def make_face_detector():
    """Use the bundled BlazeFace short-range TFLite (same as on-device)."""
    import mediapipe as mp
    model_path = str(
        Path(__file__).parent.parent /
        "app/src/main/assets/blaze_face_short_range.tflite"
    )
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        min_detection_confidence=0.4,
    )
    return mp.tasks.vision.FaceDetector.create_from_options(options)


def detect_face(detector, img_bgr: np.ndarray):
    """Return (5_keypoints_xy, det_score) or None.

    BlazeFace emits 6 keypoints per face: right_eye, left_eye, nose,
    mouth_center, right_ear, left_ear. We use the first 4 plus synthesized
    mouth corners for the ArcFace 5-point similarity transform.
    """
    import mediapipe as mp
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mp_image)
    if not res.detections:
        return None
    best = max(res.detections,
               key=lambda d: d.bounding_box.width * d.bounding_box.height)
    kp = best.keypoints
    if len(kp) < 4:
        return None
    # In Tasks API, keypoints are in normalized [0,1] coordinates by default.
    # Convert to pixels.
    h, w = img_bgr.shape[:2]
    rex, rey = kp[0].x * w, kp[0].y * h
    lex, ley = kp[1].x * w, kp[1].y * h
    nx,  ny  = kp[2].x * w, kp[2].y * h
    mcx, mcy = kp[3].x * w, kp[3].y * h
    eye_dx = lex - rex
    eye_dy = ley - rey
    half_mouth_offset = 0.35
    mr_x = mcx - eye_dx * half_mouth_offset
    mr_y = mcy - eye_dy * half_mouth_offset
    ml_x = mcx + eye_dx * half_mouth_offset
    ml_y = mcy + eye_dy * half_mouth_offset
    pts = np.array([
        [rex, rey], [lex, ley], [nx, ny], [mr_x, mr_y], [ml_x, ml_y],
    ], dtype=np.float32)
    score = best.categories[0].score if best.categories else 0.0
    return pts, score


def align_face(img_bgr: np.ndarray, kp: np.ndarray, size: int = 112):
    """Similarity transform from 5 detected keypoints to ArcFace template."""
    M, _ = cv2.estimateAffinePartial2D(kp, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    out = cv2.warpAffine(img_bgr, M, (size, size), borderValue=0)
    return out


# --------------------------------------------------------------------- models


def build_mobilefacenet():
    """The bundled MobileFaceNet TFLite has a fixed batch=2 input. The on-device
    code fills slot 0 with the face and slot 1 with zeros (FaceEmbedder.kt
    fillInputBuffer). We mirror that here and read output[0]."""
    import tensorflow as tf
    model_path = str(
        Path(__file__).parent.parent / "app/src/main/assets/mobilefacenet.tflite"
    )
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_shape = inp["shape"]   # [2, H, W, 3]
    h, w = int(in_shape[1]), int(in_shape[2])

    def embed(face_bgr_112: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr_112, cv2.COLOR_BGR2RGB)
        if (rgb.shape[0], rgb.shape[1]) != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        face_arr = (rgb.astype(np.float32) - 127.5) / 128.0
        # Stack [face, zeros] to match the fixed batch=2 model.
        zeros = np.zeros_like(face_arr)
        batch = np.stack([face_arr, zeros], axis=0)
        interp.set_tensor(inp["index"], batch.astype(np.float32))
        interp.invoke()
        feat = interp.get_tensor(out["index"])[0].reshape(-1)
        n = np.linalg.norm(feat)
        return (feat / n).astype(np.float32) if n > 0 else feat.astype(np.float32)

    return embed, (h, w)


def build_edgeface_xs():
    import torch
    model = torch.hub.load("otroshi/edgeface", "edgeface_xs_gamma_06",
                           source="github", pretrained=True, trust_repo=True)
    model.eval()

    def embed(face_bgr_112: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr_112, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != 112 or rgb.shape[1] != 112:
            rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32) / 255.0
        # EdgeFace normalize: mean=[0.5]*3, std=[0.5]*3.
        arr = (arr - 0.5) / 0.5
        # NCHW float32 tensor.
        t = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None, ...].copy())
        with torch.no_grad():
            feat = model(t).cpu().numpy().reshape(-1)
        n = np.linalg.norm(feat)
        return (feat / n).astype(np.float32) if n > 0 else feat.astype(np.float32)

    return embed, (112, 112)


MODELS = {
    "mobilefacenet":   build_mobilefacenet,
    "edgeface_xs":     build_edgeface_xs,
}


# --------------------------------------------------------------------- benchmark


def collect_aligned_faces(video_dir: Path, identity_map: dict, max_per_track: int):
    """Detect+align face per crop; return {(identity, track): [aligned_face_bgr_112]}."""
    detector = make_face_detector()
    track_to_id = {int(k): v for k, v in identity_map["tracks"].items()}
    crops = sorted(video_dir.glob("*.png"))

    by_identity = defaultdict(lambda: defaultdict(list))
    no_face = 0
    for path in crops:
        _, track = parse_filename(path.name)
        ident = track_to_id.get(track, "unknown")
        if ident == "unknown":
            continue
        if len(by_identity[ident][track]) >= max_per_track:
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        face = detect_face(detector, bgr)
        if face is None:
            no_face += 1
            continue
        kp, _score = face
        aligned = align_face(bgr, kp, size=112)
        if aligned is None:
            no_face += 1
            continue
        by_identity[ident][track].append(aligned)
    # Tasks API FaceDetector — close via __del__/automatic cleanup; .close() not exposed.
    print(f"  detected face in {sum(len(t) for tracks in by_identity.values() for t in tracks.values())} of "
          f"{len(crops)} crops (no-face: {no_face})")
    return by_identity


def run_benchmark(video_dir: Path, identity_map: dict, model_name: str,
                  max_per_track: int):
    print(f"  building {model_name}...")
    embed, _ = MODELS[model_name]()
    print(f"  collecting aligned faces...")
    aligned = collect_aligned_faces(video_dir, identity_map, max_per_track)

    aligned.pop("unknown", None)
    if len(aligned) < 2:
        print(f"  fewer than 2 named identities with detected faces — skipping")
        return

    print(f"  embedding...")
    by_identity_embs = {ident: [embed(face)
                                for tracks in t.values()
                                for face in tracks]
                        for ident, t in aligned.items()}

    same_sims = defaultdict(list)
    for ident, embs in by_identity_embs.items():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                same_sims[ident].append(cosine(embs[i], embs[j]))

    diff_sims = []
    keys = sorted(by_identity_embs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for a in by_identity_embs[keys[i]]:
                for b in by_identity_embs[keys[j]]:
                    diff_sims.append(cosine(a, b))

    all_same = [s for v in same_sims.values() for s in v]
    if not all_same or not diff_sims:
        print("  empty distributions")
        return

    same_p10 = percentile(all_same, 10)
    same_p50 = percentile(all_same, 50)
    diff_p50 = percentile(diff_sims, 50)
    diff_p90 = percentile(diff_sims, 90)
    diff_p99 = percentile(diff_sims, 99)

    print(f"  same:  n={len(all_same):<6d} p10={same_p10:+.3f} p50={same_p50:+.3f} max={max(all_same):+.3f}")
    print(f"  diff:  n={len(diff_sims):<6d} p50={diff_p50:+.3f} p90={diff_p90:+.3f} p99={diff_p99:+.3f}")
    print(f"  gap (same_p10 - diff_p90):  {same_p10 - diff_p90:+.3f}")
    print(f"  gap (same_p10 - diff_p99):  {same_p10 - diff_p99:+.3f}")

    print(f"  same per identity:")
    for ident in keys:
        sims = same_sims.get(ident, [])
        if sims:
            print(f"    {ident:<8s} n={len(sims):<5d} p10={percentile(sims,10):+.3f} p50={percentile(sims,50):+.3f}")
    print(f"  diff per identity-pair:")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ai, bi = keys[i], keys[j]
            ps = []
            for a in by_identity_embs[ai]:
                for b in by_identity_embs[bi]:
                    ps.append(cosine(a, b))
            if ps:
                print(f"    {ai} vs {bi}: n={len(ps):<5d} p10={percentile(ps,10):+.3f} p50={percentile(ps,50):+.3f} p90={percentile(ps,90):+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-root", type=Path, default=Path("crops"))
    ap.add_argument("--identity-map", type=Path,
                    default=Path("crops/identity_map.json"))
    ap.add_argument("--video", choices=["kid_to_wife", "boy_indoor_wife_swap",
                                        "man_desk", "all"], default="all")
    ap.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    ap.add_argument("--max-per-track", type=int, default=30)
    args = ap.parse_args()

    with open(args.identity_map) as f:
        imap = json.load(f)

    videos = ["kid_to_wife", "boy_indoor_wife_swap", "man_desk"] \
        if args.video == "all" else [args.video]
    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for video in videos:
        video_dir = args.crops_root / video
        if not video_dir.is_dir():
            print(f"skip {video}: no crops at {video_dir}")
            continue
        cfg = imap["videos"].get(video)
        if cfg is None:
            continue
        for model in models:
            print(f"\n==== {video}  /  {model} ====")
            run_benchmark(video_dir, cfg, model, args.max_per_track)


if __name__ == "__main__":
    main()
