#!/usr/bin/env python3
"""Compare embedding models for object identity discrimination.

Tests MobileNetV3 (current, via MediaPipe) vs DINOv2 (candidate, via torch)
on crops from debug frames to see which better distinguishes same-category objects.

Usage:
    cd tools && .venv/bin/python compare_embedders.py
"""

import numpy as np
from pathlib import Path

# --- Crop definitions from debug frames ---
# Format: (image_path, normalized_box [left, top, right, bottom], label)
FIXTURES_DIR = Path(__file__).parent / "test_fixtures" / "two_cars" / "frames"

CROPS = [
    # Dark/grey car — the one we locked on
    (FIXTURES_DIR / "190950_528_LOCK.png", [0.15, 0.35, 0.85, 0.75], "dark_car"),
    # Same dark car from different angle (reacquire frame)
    (FIXTURES_DIR / "191004_442_REACQUIRE.png", [0.1, 0.2, 0.9, 0.8], "dark_car_angle2"),
    # Red car — the wrong reacquire
    (FIXTURES_DIR / "191011_074_REACQUIRE.png", [0.3, 0.35, 0.75, 0.65], "red_car"),
]


def crop_image(img, box):
    """Crop image using normalized [0,1] box coordinates."""
    w, h = img.size
    left = int(box[0] * w)
    top = int(box[1] * h)
    right = int(box[2] * w)
    bottom = int(box[3] * h)
    return img.crop((left, top, right, bottom))


def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


def test_mobilenet():
    """Test with MediaPipe MobileNetV3 (same as Android app)."""
    print("\n=== MobileNetV3 Small (MediaPipe, current) ===")
    try:
        import mediapipe as mp
        from PIL import Image
    except ImportError:
        print("  mediapipe or Pillow not installed, skipping")
        return None

    model_path = str(
        Path(__file__).parent.parent
        / "app/src/main/assets/mobilenet_v3_small_075_224_embedder.tflite"
    )

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=True, quantize=False
    )
    embedder = mp.tasks.vision.ImageEmbedder.create_from_options(options)

    embeddings = {}
    for path, box, label in CROPS:
        img = Image.open(path).convert("RGB")
        cropped = crop_image(img, box)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(cropped))
        result = embedder.embed(mp_image)
        emb = result.embeddings[0].embedding
        embeddings[label] = emb
        print(f"  {label}: dim={len(emb)}")

    embedder.close()
    return embeddings


def test_dinov2():
    """Test with DINOv2 (candidate model)."""
    print("\n=== DINOv2 Small (torch, candidate) ===")
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
    except ImportError:
        print("  torch/torchvision not installed, skipping")
        return None

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = {}
    with torch.no_grad():
        for path, box, label in CROPS:
            img = Image.open(path).convert("RGB")
            cropped = crop_image(img, box)
            tensor = transform(cropped).unsqueeze(0)
            emb = model(tensor).squeeze().numpy()
            # L2 normalize
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings[label] = emb
            print(f"  {label}: dim={len(emb)}")

    return embeddings


def test_clip():
    """Test with CLIP ViT-B/32 (another candidate)."""
    print("\n=== CLIP ViT-B/32 (torch, candidate) ===")
    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError:
        print("  open_clip not installed, skipping")
        return None

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for path, box, label in CROPS:
            img = Image.open(path).convert("RGB")
            cropped = crop_image(img, box)
            tensor = preprocess(cropped).unsqueeze(0)
            emb = model.encode_image(tensor).squeeze().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings[label] = emb
            print(f"  {label}: dim={len(emb)}")

    return embeddings


def print_similarity_matrix(name, embeddings):
    if not embeddings:
        return
    labels = list(embeddings.keys())
    print(f"\n  Similarity matrix ({name}):")
    print(f"  {'':>20}", end="")
    for l in labels:
        print(f"  {l:>16}", end="")
    print()
    for i, l1 in enumerate(labels):
        print(f"  {l1:>20}", end="")
        for j, l2 in enumerate(labels):
            sim = cosine_sim(embeddings[l1], embeddings[l2])
            print(f"  {sim:>16.3f}", end="")
        print()

    # Key metric: same-car sim vs different-car sim
    same = cosine_sim(embeddings["dark_car"], embeddings["dark_car_angle2"])
    diff = cosine_sim(embeddings["dark_car"], embeddings["red_car"])
    gap = same - diff
    print(f"\n  Same car similarity:      {same:.3f}")
    print(f"  Different car similarity: {diff:.3f}")
    print(f"  Discrimination gap:       {gap:.3f} {'PASS' if gap > 0.1 else 'FAIL (< 0.1)'}")


if __name__ == "__main__":
    print("Comparing embedding models for same-category object discrimination")
    print("=" * 70)

    mn_emb = test_mobilenet()
    print_similarity_matrix("MobileNetV3", mn_emb)

    dino_emb = test_dinov2()
    print_similarity_matrix("DINOv2", dino_emb)

    clip_emb = test_clip()
    print_similarity_matrix("CLIP", clip_emb)

    print("\n" + "=" * 70)
    print("Done. Compare discrimination gaps — higher is better.")
