#!/usr/bin/env python3
"""Benchmark embedding model quality against annotated test scenarios.

Loads a scenario JSON + frames, runs the specified TFLite model through
MediaPipe Image Embedder (same pipeline as Android), and reports:
  - Cosine similarity matrix between all annotated object crops
  - Per-scenario re-acquisition pass/fail
  - Summary metrics for model comparison

Usage:
    python benchmark_embeddings.py <scenario.json> <frames_dir> [--model path/to/model.tflite]

The default model is the one used in the Android app.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

DEFAULT_MODEL = str(
    Path(__file__).parent.parent
    / "app/src/main/assets/mobilenet_v3_small_075_224_embedder.tflite"
)


@dataclass
class CropEmbedding:
    object_id: str
    frame: str
    box: list[float]
    embedding: np.ndarray


@dataclass
class BenchmarkResult:
    model_path: str
    scenario: str
    embeddings: list[CropEmbedding] = field(default_factory=list)
    similarity_matrix: dict = field(default_factory=dict)
    reacquisition_results: list[dict] = field(default_factory=list)


def create_embedder(model_path: str) -> mp.tasks.vision.ImageEmbedder:
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=True,
        quantize=False,
    )
    return mp.tasks.vision.ImageEmbedder.create_from_options(options)


def crop_and_embed(
    embedder: mp.tasks.vision.ImageEmbedder,
    image_path: Path,
    box: list[float],
) -> np.ndarray | None:
    """Crop a normalized bounding box from an image and compute its embedding."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    left = int(box[0] * w)
    top = int(box[1] * h)
    right = int(box[2] * w)
    bottom = int(box[3] * h)

    crop = img.crop((left, top, right, bottom))
    if crop.width < 1 or crop.height < 1:
        return None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop))
    result = embedder.embed(mp_image)
    emb = result.embeddings[0]
    return np.array(emb.embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_benchmark(scenario_path: str, frames_dir: str, model_path: str) -> BenchmarkResult:
    scenario = json.loads(Path(scenario_path).read_text())
    frames = Path(frames_dir)
    embedder = create_embedder(model_path)

    result = BenchmarkResult(
        model_path=model_path,
        scenario=scenario.get("description", scenario_path),
    )

    # Step 1: Compute embeddings for all annotated crops
    print(f"\nModel: {Path(model_path).name}")
    print(f"Scenario: {result.scenario}")
    print(f"Frames dir: {frames_dir}\n")

    for frame_ann in scenario["frames"]:
        frame_path = frames / frame_ann["frame"]
        if not frame_path.exists():
            print(f"  WARN: {frame_path} not found, skipping")
            continue

        for obj in frame_ann["objects"]:
            emb = crop_and_embed(embedder, frame_path, obj["box"])
            if emb is None:
                print(f"  WARN: Failed to embed {obj['object_id']}@{frame_ann['frame']}")
                continue

            result.embeddings.append(CropEmbedding(
                object_id=obj["object_id"],
                frame=frame_ann["frame"],
                box=obj["box"],
                embedding=emb,
            ))

    print(f"Computed {len(result.embeddings)} embeddings\n")

    # Step 2: Similarity matrix
    print("=" * 70)
    print("SIMILARITY MATRIX")
    print("=" * 70)

    labels = [f"{e.object_id}@{e.frame}" for e in result.embeddings]
    n = len(result.embeddings)

    # Print header
    max_label = max(len(l) for l in labels) if labels else 10
    header = " " * (max_label + 2) + "  ".join(f"{i:5d}" for i in range(n))
    print(header)

    for i in range(n):
        row_label = labels[i].ljust(max_label + 2)
        row_vals = []
        for j in range(n):
            sim = cosine_similarity(
                result.embeddings[i].embedding,
                result.embeddings[j].embedding,
            )
            result.similarity_matrix[f"{labels[i]} vs {labels[j]}"] = sim
            row_vals.append(f"{sim:5.3f}")
        print(f"{row_label}{'  '.join(row_vals)}")

    # Step 3: Same-object vs cross-object summary
    print(f"\n{'=' * 70}")
    print("SAME vs DIFFERENT OBJECT SIMILARITY")
    print("=" * 70)

    same_obj_sims = []
    diff_obj_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(
                result.embeddings[i].embedding,
                result.embeddings[j].embedding,
            )
            if result.embeddings[i].object_id == result.embeddings[j].object_id:
                same_obj_sims.append(sim)
            else:
                diff_obj_sims.append(sim)

    if same_obj_sims:
        print(f"Same object:      mean={np.mean(same_obj_sims):.3f}  "
              f"min={np.min(same_obj_sims):.3f}  max={np.max(same_obj_sims):.3f}  "
              f"(n={len(same_obj_sims)})")
    if diff_obj_sims:
        print(f"Different object: mean={np.mean(diff_obj_sims):.3f}  "
              f"min={np.min(diff_obj_sims):.3f}  max={np.max(diff_obj_sims):.3f}  "
              f"(n={len(diff_obj_sims)})")
    if same_obj_sims and diff_obj_sims:
        gap = np.mean(same_obj_sims) - np.mean(diff_obj_sims)
        print(f"Separation gap:   {gap:.3f}  {'GOOD' if gap > 0.15 else 'WEAK' if gap > 0.05 else 'POOR'}")

    # Step 4: Evaluate expected re-acquisitions
    print(f"\n{'=' * 70}")
    print("RE-ACQUISITION SCENARIOS")
    print("=" * 70)

    for expected in scenario.get("expected_reacquisitions", []):
        desc = expected["description"]
        lock = expected["lock_on"]
        target = expected["should_reacquire"]
        distractors = expected.get("should_not_reacquire", [])

        # Find the lock embedding
        lock_emb = _find_embedding(result.embeddings, lock["object_id"], lock["frame"])
        if lock_emb is None:
            print(f"  SKIP: {desc} — lock embedding not found")
            continue

        # Find target embedding
        target_emb = _find_embedding(result.embeddings, target["object_id"], target["frame"])
        if target_emb is None:
            print(f"  SKIP: {desc} — target embedding not found")
            continue

        target_sim = cosine_similarity(lock_emb.embedding, target_emb.embedding)

        # Check against distractors
        passed = True
        details = [f"target={target_sim:.3f}"]

        for distractor in distractors:
            dist_emb = _find_embedding(
                result.embeddings,
                distractor["object_id"],
                distractor["frame"],
            )
            if dist_emb is None:
                continue
            dist_sim = cosine_similarity(lock_emb.embedding, dist_emb.embedding)
            details.append(f"distractor({distractor['object_id']})={dist_sim:.3f}")
            if dist_sim >= target_sim:
                passed = False

        status = "PASS" if passed else "FAIL"
        result.reacquisition_results.append({
            "description": desc,
            "passed": passed,
            "target_similarity": target_sim,
            "details": details,
        })
        print(f"  [{status}] {desc}")
        print(f"         {', '.join(details)}")

    embedder.close()
    return result


def _find_embedding(
    embeddings: list[CropEmbedding], object_id: str, frame: str
) -> CropEmbedding | None:
    for e in embeddings:
        if e.object_id == object_id and e.frame == frame:
            return e
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark embedding model quality")
    parser.add_argument("scenario", help="Path to scenario JSON")
    parser.add_argument("frames_dir", help="Directory containing frame PNGs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .tflite model")
    args = parser.parse_args()

    run_benchmark(args.scenario, args.frames_dir, args.model)
