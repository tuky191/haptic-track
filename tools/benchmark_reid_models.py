#!/usr/bin/env python3
"""Same-vs-different person cosine benchmark.

Runs one or more re-identification embedders on the track-tagged person
crops produced by ``extract_person_crops.py`` (under ``crops/<video>/``)
and reports the gap between same-identity and different-identity cosine
distributions.

Identity ground truth comes from ``crops/identity_map.json``: each track ID
is hand-labeled with a person name (boy / wife / man / unknown). Crops in
the same name within a video → same identity. Crops across names within a
video → different identity. Crops labeled "unknown" are dropped from the
comparison so they neither inflate nor deflate the gap.

The metric we care about:

  same_p10 - other_p90

Positive = the model has a working threshold separating same-vs-different
at this device's image distribution. Negative = signals overlap, no fixed
threshold works (this is the kid_to_wife failure mode of the current
OSNet x1.0 — see #102).

Usage:
    cd tools && ../.venv/bin/python benchmark_reid_models.py \
        --crops-root crops --identity-map crops/identity_map.json \
        --video kid_to_wife
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


# --------------------------------------------------------------------- helpers


def load_identity_map(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_filename(name: str) -> tuple[int, int]:
    """frame_NNNNN_track_TTT.png → (frame, track)."""
    stem = Path(name).stem
    parts = stem.split("_")
    return int(parts[1]), int(parts[3])


def list_crops(video_dir: Path) -> list[tuple[Path, int, int]]:
    out = []
    for p in sorted(video_dir.glob("*.png")):
        frame, track = parse_filename(p.name)
        out.append((p, frame, track))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(xs, p))


# --------------------------------------------------------------------- models


def build_mobilenet_embedder():
    """Current generic appearance embedder (MobileNetV3 Large via MediaPipe)."""
    import mediapipe as mp
    model_path = str(
        Path(__file__).parent.parent
        / "app/src/main/assets/mobilenet_v3_large_embedder.tflite"
    )
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=True, quantize=False,
    )
    embedder = mp.tasks.vision.ImageEmbedder.create_from_options(options)

    def embed(img_path: Path) -> np.ndarray:
        img = mp.Image.create_from_file(str(img_path))
        result = embedder.embed(img)
        return np.array(result.embeddings[0].embedding, dtype=np.float32)

    return embed


def _resize_letterbox(img: Image.Image, target_w: int, target_h: int,
                     pad: tuple[int, int, int] = (114, 114, 114)) -> Image.Image:
    """Aspect-preserving resize with gray padding to (target_w, target_h)."""
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), pad)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def build_osnet_embedder(model_path: str | None = None):
    """OSNet via TFLite. Defaults to the bundled osnet_x1_0_market shipping
    model. Pass `model_path` to benchmark a different TFLite variant."""
    import tensorflow as tf
    if model_path is None:
        model_path = str(
            Path(__file__).parent.parent / "app/src/main/assets/osnet_x1_0_market.tflite"
        )
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_shape = input_details[0]["shape"]      # [1, H, W, 3] or [1, 3, H, W]
    nhwc = in_shape[-1] == 3
    h_idx, w_idx = (1, 2) if nhwc else (2, 3)
    h = int(in_shape[h_idx])
    w = int(in_shape[w_idx])

    def embed(img_path: Path) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        img = _resize_letterbox(img, w, h)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # ImageNet stats — match what the Android pipeline does for OSNet.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        if nhwc:
            arr = arr[None, ...]
        else:
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
        interpreter.set_tensor(input_details[0]["index"], arr.astype(np.float32))
        interpreter.invoke()
        feat = interpreter.get_tensor(output_details[0]["index"]).reshape(-1)
        # L2 normalize — matches the Android side.
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        return feat.astype(np.float32)

    return embed


def build_torchreid_osnet(arch: str, weights: Path, input_hw=(256, 128),
                          resize_mode: str = "letterbox"):
    """Run a torchreid OSNet variant on PyTorch. We bypass the classifier head
    and read the model's `featuremaps` (post-pool feature vector) directly.

    `resize_mode`: "letterbox" matches the on-device CanonicalCropper (#100);
    "stretch" matches torchreid's training transform (Resize((H, W)) — direct
    resize, no aspect preservation). The training-matched mode tells us the
    model's intrinsic ceiling; the deployment-matched mode tells us what we'd
    actually ship with."""
    import torch
    from torchreid.reid.models import build_model
    from torchreid.reid.utils import load_pretrained_weights

    model = build_model(name=arch, num_classes=1041, pretrained=False)
    load_pretrained_weights(model, str(weights))
    model.eval()

    h, w = input_hw

    def embed(img_path: Path) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        if resize_mode == "letterbox":
            img = _resize_letterbox(img, w, h)
        elif resize_mode == "stretch":
            img = img.resize((w, h), Image.BILINEAR)
        else:
            raise ValueError(f"unknown resize_mode {resize_mode}")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        t = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None, ...].copy())
        with torch.no_grad():
            feat = model(t).cpu().numpy().reshape(-1)
        n = np.linalg.norm(feat)
        return (feat / n).astype(np.float32) if n > 0 else feat.astype(np.float32)

    return embed


_MODELS_DIR = Path(__file__).parent / "models/osnet_hf"


def _osnet_pair(label: str, arch: str, ckpt_name: str) -> dict:
    """Return {<label>_<lb|st>: builder} pair for letterbox + stretch modes.

    `label` is the dict key prefix (e.g. "osnet_x1_0_msmt17"); `arch` is the
    torchreid model name passed to build_model (e.g. "osnet_x1_0")."""
    weights = _MODELS_DIR / ckpt_name
    return {
        f"{label}_lb": (lambda: build_torchreid_osnet(arch, weights, resize_mode="letterbox")),
        f"{label}_st": (lambda: build_torchreid_osnet(arch, weights, resize_mode="stretch")),
    }


_TFLITE_DIR = Path(__file__).parent / "models/tflite/osnet_ibn_x1_0_msmt17"


MODELS: dict[str, Callable[[], Callable[[Path], np.ndarray]]] = {
    "mobilenet_v3_large":   build_mobilenet_embedder,
    "osnet_x1_0_market":    build_osnet_embedder,
    **_osnet_pair("osnet_x1_0_msmt17",     "osnet_x1_0",     "osnet_x1_0_msmt17_combineall.pth"),
    **_osnet_pair("osnet_ain_x1_0_msmt17", "osnet_ain_x1_0", "osnet_ain_x1_0_msmt17.pth"),
    **_osnet_pair("osnet_ibn_x1_0_msmt17", "osnet_ibn_x1_0", "osnet_ibn_x1_0_msmt17_combineall.pth"),
    "osnet_ibn_x1_0_msmt17_tflite_fp32":
        lambda: build_osnet_embedder(str(_TFLITE_DIR / "osnet_ibn_x1_0_msmt17_float32.tflite")),
    "osnet_ibn_x1_0_msmt17_tflite_fp16":
        lambda: build_osnet_embedder(str(_TFLITE_DIR / "osnet_ibn_x1_0_msmt17_float16.tflite")),
}


# --------------------------------------------------------------------- benchmark


def run_benchmark(video_dir: Path, identity_map: dict, model_name: str):
    embedder = MODELS[model_name]()

    track_to_id = {int(k): v for k, v in identity_map["tracks"].items()}
    crops = list_crops(video_dir)
    if not crops:
        print(f"  no crops found in {video_dir}", file=sys.stderr)
        return

    # Embed every crop, group by (identity, track).
    by_identity: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for path, _frame, track in crops:
        ident = track_to_id.get(track, "unknown")
        emb = embedder(path)
        by_identity[ident][track].append(emb)

    # Drop "unknown" identity from comparison.
    by_identity.pop("unknown", None)

    if len(by_identity) < 2:
        print(f"  fewer than 2 named identities; skipping comparison")
        return

    # Build same-identity and different-identity sim distributions.
    same_sims: dict[str, list[float]] = defaultdict(list)
    diff_sims: list[float] = []

    identities = sorted(by_identity.keys())
    # Within-identity: pair every embedding with every later embedding from the same identity
    for ident, tracks in by_identity.items():
        flat = []
        for t, embs in tracks.items():
            for e in embs:
                flat.append(e)
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                same_sims[ident].append(cosine(flat[i], flat[j]))

    # Cross-identity: every pair across distinct identity buckets
    flats: dict[str, list[np.ndarray]] = {
        ident: [e for embs in tracks.values() for e in embs]
        for ident, tracks in by_identity.items()
    }
    keys = list(flats.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for a in flats[keys[i]]:
                for b in flats[keys[j]]:
                    diff_sims.append(cosine(a, b))

    # Combined same-sims for the gap metric.
    all_same = [s for sims in same_sims.values() for s in sims]
    if not all_same or not diff_sims:
        print("  empty same or diff distribution")
        return

    same_p10 = percentile(all_same, 10)
    same_p50 = percentile(all_same, 50)
    diff_p50 = percentile(diff_sims, 50)
    diff_p90 = percentile(diff_sims, 90)
    diff_p99 = percentile(diff_sims, 99)
    gap_p10_p90 = same_p10 - diff_p90
    gap_p10_p99 = same_p10 - diff_p99

    print(f"  same:  n={len(all_same):<6d} p10={same_p10:+.3f} p50={same_p50:+.3f} max={max(all_same):+.3f}")
    print(f"  diff:  n={len(diff_sims):<6d} p50={diff_p50:+.3f} p90={diff_p90:+.3f} p99={diff_p99:+.3f}")
    print(f"  gap (same_p10 - diff_p90):  {gap_p10_p90:+.3f}")
    print(f"  gap (same_p10 - diff_p99):  {gap_p10_p99:+.3f}")

    # Per-identity-pair breakdown for context.
    print(f"  same per identity:")
    for ident in identities:
        sims = same_sims.get(ident, [])
        if sims:
            print(f"    {ident:<8s} n={len(sims):<5d} p10={percentile(sims,10):+.3f} p50={percentile(sims,50):+.3f}")
    print(f"  diff per identity-pair:")
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            ai, bi = identities[i], identities[j]
            ps = []
            for a in flats[ai]:
                for b in flats[bi]:
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
    args = ap.parse_args()

    imap = load_identity_map(args.identity_map)
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
            print(f"skip {video}: no identity map entry")
            continue
        for model in models:
            print(f"\n==== {video}  /  {model} ====")
            run_benchmark(video_dir, cfg, model)


if __name__ == "__main__":
    main()
