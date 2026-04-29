#!/usr/bin/env python3
"""Confirm whether the bundled OSNet-IBN MSMT17 TFLite is broken at the model
level (Python CPU inference) or only fails through Android's GPU delegate.

Prints the first 4 floats of the L2-normalized output for each crop, plus
the pairwise cosine matrix. A discriminative model produces varied output
and pairwise cosines well below 1.0; a degenerate model produces identical
output and cosines pinned to 1.0.

Run: tools/.venv/bin/python tools/test_osnet_tflite_cpu.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


REPO = Path(__file__).resolve().parent.parent
CROPS = sorted((REPO / "tools" / "crops" / "man_desk").glob("*.png"))[:6]

MODELS = {
    "osnet_x1_0_market (working baseline)":
        REPO / "app/src/main/assets/osnet_x1_0_market.tflite",
    "osnet_ibn_x1_0_msmt17 fp16 (shipped in #113)":
        REPO / "app/src/main/assets/osnet_ibn_x1_0_msmt17.tflite",
    "osnet_ibn_x1_0_msmt17 fp32 (pre-quant)":
        REPO / "tools/models/tflite/osnet_ibn_x1_0_msmt17/osnet_ibn_x1_0_msmt17_float32.tflite",
}


def letterbox(img: Image.Image, w: int, h: int) -> Image.Image:
    sw, sh = img.size
    s = min(w / sw, h / sh)
    nw, nh = max(1, int(sw * s)), max(1, int(sh * s))
    canvas = Image.new("RGB", (w, h), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
    return canvas


def embed_one(interp, in_det, out_det, h, w, nhwc, img_path: Path) -> np.ndarray:
    img = letterbox(Image.open(img_path).convert("RGB"), w, h)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = arr[None, ...] if nhwc else np.transpose(arr, (2, 0, 1))[None, ...]
    interp.set_tensor(in_det[0]["index"], arr.astype(np.float32))
    interp.invoke()
    feat = interp.get_tensor(out_det[0]["index"]).reshape(-1)
    n = np.linalg.norm(feat)
    return (feat / n) if n > 0 else feat


def benchmark(label: str, model_path: Path):
    print(f"\n=== {label} ===")
    print(f"   {model_path}")
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_shape = in_det[0]["shape"]
    nhwc = in_shape[-1] == 3
    h_idx, w_idx = (1, 2) if nhwc else (2, 3)
    h, w = int(in_shape[h_idx]), int(in_shape[w_idx])
    print(f"   input: {tuple(in_shape)} {'NHWC' if nhwc else 'NCHW'}  {w}x{h}")
    print(f"   output: {tuple(out_det[0]['shape'])}\n")

    feats = []
    for p in CROPS:
        f = embed_one(interp, in_det, out_det, h, w, nhwc, p)
        feats.append(f)
        print(f"   {p.name}  out=[{f[0]:+.4f},{f[1]:+.4f},{f[2]:+.4f},{f[3]:+.4f},...]  "
              f"min={f.min():+.4f} max={f.max():+.4f} nonzero={int((f != 0).sum())}/{len(f)}")

    print("\n   Pairwise cosine (rows × cols, same crops in same order):")
    F = np.stack(feats)
    C = F @ F.T
    for i in range(len(feats)):
        print("   " + " ".join(f"{C[i, j]:+.3f}" for j in range(len(feats))))


def main():
    print(f"Crops: {len(CROPS)} from {CROPS[0].parent if CROPS else '?'}")
    if not CROPS:
        raise SystemExit("No crops found — run tools/extract_person_crops.py first")
    for label, path in MODELS.items():
        if not path.exists():
            print(f"\n=== {label} ===  MISSING {path}")
            continue
        benchmark(label, path)


if __name__ == "__main__":
    main()
