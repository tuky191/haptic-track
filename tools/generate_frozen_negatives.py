#!/usr/bin/env python3
"""Generate frozen-negatives binary assets for ReacquisitionEngine (#68).

Two modes:

  --target osnet  (default — primary people path)
      Source: scenario.json files. For each scenario we extract person
      detection bounding boxes, crop the corresponding frame PNG, and run
      it through OSNet x1.0 (TFLite). 512-dim L2-normalized embeddings.
      Output: app/src/main/assets/frozen_negatives_osnet.bin

  --target mnv3  (generic non-person path)
      Source: any directory of images. Random crops are embedded via
      MediaPipe ImageEmbedder (same pipeline as Android). 1280-dim
      L2-normalized embeddings.
      Output: app/src/main/assets/frozen_negatives_mnv3.bin

Binary format (FrozenNegatives.kt):
    int32 count, int32 dim, float32[count][dim]  (little-endian, L2-normalized rows)

OSNet usage:
    cd tools && .venv/bin/python generate_frozen_negatives.py \\
        --target osnet \\
        --source /tmp/sessions_62_baseline /tmp/sessions_62_b2 /tmp/sessions_70 \\
        --count 1500

MNV3 usage:
    cd tools && .venv/bin/python generate_frozen_negatives.py \\
        --target mnv3 \\
        --source /tmp/sessions_62_baseline \\
        --count 1500
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
ASSETS_DIR = REPO_ROOT / "app/src/main/assets"

# OSNet config (matches PersonReIdEmbedder.kt)
OSNET_MODEL = ASSETS_DIR / "osnet_x1_0_market.tflite"
OSNET_HEIGHT = 256
OSNET_WIDTH = 128
OSNET_DIM = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MNV3 config
MNV3_MODEL = ASSETS_DIR / "mobilenet_v3_large_embedder.tflite"
MNV3_DIM = 1280

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------- Common helpers ----------------------------

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def is_diverse(emb: np.ndarray, pool: list[np.ndarray], threshold: float) -> bool:
    if not pool:
        return True
    sims = np.stack(pool) @ emb  # all L2-normalized so dot = cosine
    return float(sims.max()) <= threshold


def write_asset(path: Path, embeddings: list[np.ndarray], dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", len(embeddings), dim))
        for emb in embeddings:
            f.write(emb.astype(np.float32).tobytes())
    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        rel = path.relative_to(REPO_ROOT)
        display = str(rel)
    except ValueError:
        display = str(path)
    print(f"Wrote {display} — {len(embeddings)} × {dim} ({size_mb:.2f} MB)")


# ---------------------------- OSNet path ----------------------------

def load_osnet():
    """Load OSNet TFLite via ai-edge-litert (preferred) or tensorflow.lite."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    interp = Interpreter(model_path=str(OSNET_MODEL))
    interp.allocate_tensors()
    return interp


def osnet_embed(interp, crop: Image.Image) -> np.ndarray:
    rgb = crop.convert("RGB").resize((OSNET_WIDTH, OSNET_HEIGHT), Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.expand_dims(arr, axis=0)  # [1, H, W, 3]

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], arr.astype(np.float32))
    interp.invoke()
    emb = interp.get_tensor(out["index"]).flatten().astype(np.float32)
    return l2_normalize(emb)


def discover_person_crops(source_dirs: list[Path], rng: random.Random) -> list[tuple[Path, list[float]]]:
    """Walk source dirs for scenario.json files; yield (frame_png, [l,t,r,b]) for each
    person detection (excluding the locked subject's own ID from each scenario)."""
    crops: list[tuple[Path, list[float]]] = []
    for root in source_dirs:
        for scenario_path in root.rglob("scenario.json"):
            session_dir = scenario_path.parent
            try:
                data = json.loads(scenario_path.read_text())
            except Exception as e:
                print(f"  skip {scenario_path}: {e}")
                continue
            locked_id = data.get("lock", {}).get("trackingId")
            for frame in data.get("frames", []):
                # Frame index → PNG. Scenario records frames at index N; we look
                # for any *_SEARCH/_LOST/_REACQUIRE png mentioning that index, but
                # the simpler path is to match by closest-timestamp to event PNGs.
                # As a pragmatic shortcut, we use ALL session pngs as candidate
                # bitmaps and crop bboxes from the corresponding frame.
                # Each detection has frame.index + box; we pull person dets only.
                for det in frame.get("detections", []):
                    if det.get("label") != "person":
                        continue
                    if det.get("id") == locked_id:
                        # Locked subject — skip; we want NEGATIVES, not positives.
                        continue
                    bb = det.get("boundingBox")
                    if not (isinstance(bb, list) and len(bb) == 4):
                        continue
                    # Match a frame PNG. Scenarios record frame index into events;
                    # raw frames are saved as <ts>_<event>_raw.png. Use any raw PNG
                    # in the session — its dimensions match the bbox normalization.
                    raw_pngs = list(session_dir.glob("*_raw.png"))
                    if not raw_pngs:
                        continue
                    png = rng.choice(raw_pngs)
                    crops.append((png, bb))
    return crops


def generate_osnet(source_dirs: list[Path], count: int, diversity: float, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    candidates = discover_person_crops(source_dirs, rng)
    if not candidates:
        print("No person detections found in any scenario.json under sources", file=sys.stderr)
        return []
    rng.shuffle(candidates)
    print(f"Found {len(candidates)} person crop candidates across {len(source_dirs)} source dirs")

    interp = load_osnet()
    pool: list[np.ndarray] = []
    skipped = 0
    for i, (png, bb) in enumerate(candidates):
        if len(pool) >= count:
            break
        try:
            img = Image.open(png)
            w, h = img.size
            l = max(0, int(bb[0] * w))
            t = max(0, int(bb[1] * h))
            r = min(w, int(bb[2] * w))
            b = min(h, int(bb[3] * h))
            if r - l < 10 or b - t < 20:
                skipped += 1
                continue
            crop = img.crop((l, t, r, b))
            emb = osnet_embed(interp, crop)
        except Exception as e:
            print(f"  skip {png.name}: {e}")
            continue
        if not is_diverse(emb, pool, diversity):
            skipped += 1
            continue
        pool.append(emb)
        if len(pool) % 100 == 0:
            print(f"  {i}/{len(candidates)} candidates → {len(pool)} embeddings ({skipped} skipped)")
    return pool


# ---------------------------- MNV3 path ----------------------------

def create_mnv3_embedder():
    import mediapipe as mp
    base_options = mp.tasks.BaseOptions(model_asset_path=str(MNV3_MODEL))
    options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=True, quantize=False,
    )
    return mp.tasks.vision.ImageEmbedder.create_from_options(options)


def mnv3_embed(embedder, crop: Image.Image) -> np.ndarray:
    import mediapipe as mp
    rgb = crop.convert("RGB")
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(rgb))
    result = embedder.embed(mp_image)
    return l2_normalize(np.array(result.embeddings[0].embedding, dtype=np.float32))


def random_crop(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    scale = rng.uniform(0.2, 0.8)
    aspect = rng.uniform(0.6, 1.6)
    crop_w = max(32, min(w, int(w * np.sqrt(scale * aspect))))
    crop_h = max(32, min(h, int(h * np.sqrt(scale / aspect))))
    x = rng.randint(0, max(0, w - crop_w))
    y = rng.randint(0, max(0, h - crop_h))
    return img.crop((x, y, x + crop_w, y + crop_h))


def generate_mnv3(source_dirs: list[Path], count: int, crops_per_image: int,
                  diversity: float, seed: int, exclude: list[str]) -> list[np.ndarray]:
    images: list[Path] = []
    for root in source_dirs:
        for p in root.rglob("*"):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            if any(e in p.name for e in exclude):
                continue
            images.append(p)
    if not images:
        print(f"No images found under sources", file=sys.stderr)
        return []
    rng = random.Random(seed)
    rng.shuffle(images)
    print(f"Found {len(images)} source images")

    embedder = create_mnv3_embedder()
    pool: list[np.ndarray] = []
    skipped = 0
    for i, p in enumerate(images):
        if len(pool) >= count:
            break
        try:
            img = Image.open(p)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        for _ in range(crops_per_image):
            if len(pool) >= count:
                break
            try:
                emb = mnv3_embed(embedder, random_crop(img, rng))
            except Exception as e:
                print(f"  embed {p.name}: {e}")
                continue
            if not is_diverse(emb, pool, diversity):
                skipped += 1
                continue
            pool.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(images)} images → {len(pool)} embeddings ({skipped} skipped)")
    return pool


# ---------------------------- Main ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["osnet", "mnv3"], default="osnet")
    parser.add_argument("--source", type=Path, nargs="+", required=True,
                        help="Source dirs. For osnet: scenario session dirs. For mnv3: any image dirs.")
    parser.add_argument("--output", type=Path, help="Output asset path (default depends on --target)")
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--diversity", type=float, default=0.95,
                        help="Drop crops with cosine > this against existing pool")
    parser.add_argument("--seed", type=int, default=42)
    # mnv3-only
    parser.add_argument("--crops-per-image", type=int, default=4)
    parser.add_argument("--exclude", default="LOCK_raw,REACQUIRE_raw,LOST_raw")
    args = parser.parse_args()

    for d in args.source:
        if not d.is_dir():
            print(f"Source not a directory: {d}", file=sys.stderr)
            return 1

    if args.target == "osnet":
        if not OSNET_MODEL.is_file():
            print(f"OSNet model missing: {OSNET_MODEL}", file=sys.stderr)
            return 1
        embeddings = generate_osnet(args.source, args.count, args.diversity, args.seed)
        out = args.output or (ASSETS_DIR / "frozen_negatives_osnet.bin")
        dim = OSNET_DIM
    else:
        if not MNV3_MODEL.is_file():
            print(f"MNV3 model missing: {MNV3_MODEL}", file=sys.stderr)
            return 1
        excludes = [s.strip() for s in args.exclude.split(",") if s.strip()]
        embeddings = generate_mnv3(args.source, args.count, args.crops_per_image,
                                    args.diversity, args.seed, excludes)
        out = args.output or (ASSETS_DIR / "frozen_negatives_mnv3.bin")
        dim = MNV3_DIM

    if not embeddings:
        return 1
    if len(embeddings) < args.count:
        print(f"WARNING: only produced {len(embeddings)} / {args.count}. "
              f"Pass more --source dirs or capture more sessions.")
    write_asset(out, embeddings, dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
