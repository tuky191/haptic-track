#!/usr/bin/env python3
"""Research benchmark: compare embedding models for object identity discrimination.

Tests multiple models against annotated test fixtures to find the best embedder
for same-category object discrimination (the core weakness identified in #50).

Models tested:
  - MobileNetV3 Large (current, via MediaPipe) — 1280-dim
  - DINOv2-small (ViT-S/14) — 384-dim
  - DINOv2-base (ViT-B/14) — 768-dim
  - DINOv2-small with registers — 384-dim
  - CLIP ViT-B/32 — 512-dim

Usage:
    cd tools && .venv/bin/python research_embeddings.py [--fixtures-dir test_fixtures]
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# ── Data classes ──

@dataclass
class CropResult:
    object_id: str
    frame: str
    embedding: np.ndarray
    embed_time_ms: float


@dataclass
class ModelResult:
    name: str
    dim: int
    crops: list[CropResult] = field(default_factory=list)
    avg_embed_ms: float = 0.0
    same_obj_sims: list[float] = field(default_factory=list)
    diff_obj_sims: list[float] = field(default_factory=list)
    reacq_results: list[dict] = field(default_factory=list)


# ── Helpers ──

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def crop_image(img: Image.Image, box: list[float]) -> Image.Image:
    w, h = img.size
    left = int(box[0] * w)
    top = int(box[1] * h)
    right = int(box[2] * w)
    bottom = int(box[3] * h)
    return img.crop((left, top, right, bottom))


def load_scenario(scenario_path: Path) -> tuple[dict, Path]:
    scenario = json.loads(scenario_path.read_text())
    frames_dir = scenario_path.parent / "frames"
    return scenario, frames_dir


# ── Model wrappers ──

class MobileNetV3Embedder:
    """MediaPipe MobileNetV3 Large — same as Android app."""

    def __init__(self):
        import mediapipe as mp
        model_path = str(
            Path(__file__).parent.parent
            / "app/src/main/assets/mobilenet_v3_large_embedder.tflite"
        )
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.ImageEmbedderOptions(
            base_options=base_options, l2_normalize=True, quantize=False
        )
        self.embedder = mp.tasks.vision.ImageEmbedder.create_from_options(options)
        self.mp = mp

    @property
    def name(self):
        return "MobileNetV3-Large"

    @property
    def dim(self):
        return 1280

    def embed(self, crop: Image.Image) -> np.ndarray:
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=np.array(crop)
        )
        result = self.embedder.embed(mp_image)
        return np.array(result.embeddings[0].embedding, dtype=np.float32)

    def close(self):
        self.embedder.close()


class DINOv2Embedder:
    """DINOv2 via torch hub."""

    def __init__(self, variant="dinov2_vits14"):
        import torch
        from torchvision import transforms

        self.torch = torch
        self.variant = variant
        self.model = torch.hub.load(
            "facebookresearch/dinov2", variant, pretrained=True
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Determine dim from a dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.model(dummy)
            self._dim = out.shape[-1]

    @property
    def name(self):
        label_map = {
            "dinov2_vits14": "DINOv2-S/14",
            "dinov2_vitb14": "DINOv2-B/14",
            "dinov2_vits14_reg": "DINOv2-S/14-reg",
            "dinov2_vitb14_reg": "DINOv2-B/14-reg",
        }
        return label_map.get(self.variant, self.variant)

    @property
    def dim(self):
        return self._dim

    def embed(self, crop: Image.Image) -> np.ndarray:
        tensor = self.transform(crop).unsqueeze(0)
        with self.torch.no_grad():
            emb = self.model(tensor).squeeze().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)

    def close(self):
        pass


class CLIPEmbedder:
    """CLIP ViT-B/32 via open_clip."""

    def __init__(self):
        import torch
        import open_clip

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.eval()

    @property
    def name(self):
        return "CLIP-ViT-B/32"

    @property
    def dim(self):
        return 512

    def embed(self, crop: Image.Image) -> np.ndarray:
        tensor = self.preprocess(crop).unsqueeze(0)
        with self.torch.no_grad():
            emb = self.model.encode_image(tensor).squeeze().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)

    def close(self):
        pass


# ── Benchmark logic ──

def benchmark_model(embedder, scenario: dict, frames_dir: Path) -> ModelResult:
    result = ModelResult(name=embedder.name, dim=embedder.dim)
    times = []

    for frame_ann in scenario["frames"]:
        frame_path = frames_dir / frame_ann["frame"]
        if not frame_path.exists():
            continue

        img = Image.open(frame_path).convert("RGB")

        for obj in frame_ann["objects"]:
            crop = crop_image(img, obj["box"])
            if crop.width < 1 or crop.height < 1:
                continue

            t0 = time.perf_counter()
            emb = embedder.embed(crop)
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)

            result.crops.append(CropResult(
                object_id=obj["object_id"],
                frame=frame_ann["frame"],
                embedding=emb,
                embed_time_ms=dt,
            ))

    result.avg_embed_ms = np.mean(times) if times else 0

    # Compute same vs different similarities
    n = len(result.crops)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_sim(result.crops[i].embedding, result.crops[j].embedding)
            if result.crops[i].object_id == result.crops[j].object_id:
                result.same_obj_sims.append(sim)
            else:
                result.diff_obj_sims.append(sim)

    # Evaluate reacquisition scenarios
    for expected in scenario.get("expected_reacquisitions", []):
        lock = expected["lock_on"]
        target = expected["should_reacquire"]
        distractors = expected.get("should_not_reacquire", [])

        lock_crop = _find_crop(result.crops, lock["object_id"], lock["frame"])
        target_crop = _find_crop(result.crops, target["object_id"], target["frame"])
        if not lock_crop or not target_crop:
            continue

        target_sim = cosine_sim(lock_crop.embedding, target_crop.embedding)
        passed = True
        dist_sims = {}

        for dist in distractors:
            dist_crop = _find_crop(result.crops, dist["object_id"], dist["frame"])
            if not dist_crop:
                continue
            dist_sim = cosine_sim(lock_crop.embedding, dist_crop.embedding)
            dist_sims[dist["object_id"]] = dist_sim
            if dist_sim >= target_sim:
                passed = False

        result.reacq_results.append({
            "description": expected["description"],
            "passed": passed,
            "target_sim": target_sim,
            "distractor_sims": dist_sims,
        })

    return result


def _find_crop(crops: list[CropResult], object_id: str, frame: str) -> CropResult | None:
    for c in crops:
        if c.object_id == object_id and c.frame == frame:
            return c
    return None


# ── Reporting ──

def print_model_result(result: ModelResult, scenario_name: str):
    print(f"\n  {result.name} ({result.dim}-dim, avg {result.avg_embed_ms:.1f}ms/crop)")

    # Similarity matrix
    n = len(result.crops)
    if n > 0:
        labels = [f"{c.object_id}@{c.frame.split('.')[0][-4:]}" for c in result.crops]
        max_label = max(len(l) for l in labels)

        print(f"    {'':>{max_label}}", end="")
        for l in labels:
            print(f"  {l:>12}", end="")
        print()

        for i in range(n):
            print(f"    {labels[i]:>{max_label}}", end="")
            for j in range(n):
                sim = cosine_sim(result.crops[i].embedding, result.crops[j].embedding)
                print(f"  {sim:>12.3f}", end="")
            print()

    # Same vs different summary
    if result.same_obj_sims:
        print(f"    Same-object:  mean={np.mean(result.same_obj_sims):.3f}  "
              f"min={np.min(result.same_obj_sims):.3f}  max={np.max(result.same_obj_sims):.3f}")
    if result.diff_obj_sims:
        print(f"    Diff-object:  mean={np.mean(result.diff_obj_sims):.3f}  "
              f"min={np.min(result.diff_obj_sims):.3f}  max={np.max(result.diff_obj_sims):.3f}")
    if result.same_obj_sims and result.diff_obj_sims:
        gap = np.mean(result.same_obj_sims) - np.mean(result.diff_obj_sims)
        verdict = "GOOD" if gap > 0.15 else "WEAK" if gap > 0.05 else "POOR"
        print(f"    Gap: {gap:+.3f} ({verdict})")
    elif result.same_obj_sims:
        print(f"    (no cross-object pairs)")

    # Reacquisition
    for r in result.reacq_results:
        status = "PASS" if r["passed"] else "FAIL"
        dist_str = ", ".join(f"{k}={v:.3f}" for k, v in r["distractor_sims"].items())
        print(f"    [{status}] {r['description']}: target={r['target_sim']:.3f} {dist_str}")


def print_comparison_table(all_results: dict[str, list[ModelResult]]):
    """Print a summary comparison across all models and scenarios."""
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    # Collect model names from first scenario
    model_names = []
    for results in all_results.values():
        model_names = [r.name for r in results]
        break

    header = f"{'Model':<20} {'Dim':>4} {'ms/crop':>7}"
    for scenario_name in all_results:
        header += f"  {'Gap':>6} {'Reacq':>5}"
    header += f"  {'AvgGap':>6}"
    print(header)
    print("-" * len(header))

    for model_idx, model_name in enumerate(model_names):
        row = ""
        gaps = []
        for scenario_name, results in all_results.items():
            if model_idx < len(results):
                r = results[model_idx]
                if not row:
                    row = f"{r.name:<20} {r.dim:>4} {r.avg_embed_ms:>6.1f}ms"

                if r.same_obj_sims and r.diff_obj_sims:
                    gap = np.mean(r.same_obj_sims) - np.mean(r.diff_obj_sims)
                    gaps.append(gap)
                else:
                    gap = np.mean(r.same_obj_sims) if r.same_obj_sims else 0
                    gaps.append(gap)

                reacq_pass = sum(1 for x in r.reacq_results if x["passed"])
                reacq_total = len(r.reacq_results)
                reacq_str = f"{reacq_pass}/{reacq_total}" if reacq_total > 0 else "n/a"
                row += f"  {gap:>+6.3f} {reacq_str:>5}"

        avg_gap = np.mean(gaps) if gaps else 0
        row += f"  {avg_gap:>+6.3f}"
        print(row)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Research: compare embedding models")
    parser.add_argument("--fixtures-dir", default="test_fixtures",
                        help="Directory containing test fixture subdirs")
    parser.add_argument("--models", nargs="*",
                        default=["mobilenet", "dinov2_s", "dinov2_b", "dinov2_s_reg", "clip"],
                        help="Models to test")
    args = parser.parse_args()

    fixtures_dir = Path(args.fixtures_dir)

    # Discover scenarios
    scenarios = {}
    for scenario_json in sorted(fixtures_dir.glob("*/scenario.json")):
        frames_dir = scenario_json.parent / "frames"
        if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
            print(f"SKIP {scenario_json.parent.name}: no frames")
            continue
        scenario, _ = load_scenario(scenario_json)
        scenarios[scenario_json.parent.name] = (scenario, frames_dir)

    if not scenarios:
        print("No scenarios with frames found!")
        return

    print(f"Found {len(scenarios)} scenarios: {', '.join(scenarios.keys())}")

    # Build model list
    model_builders = {
        "mobilenet": lambda: MobileNetV3Embedder(),
        "dinov2_s": lambda: DINOv2Embedder("dinov2_vits14"),
        "dinov2_b": lambda: DINOv2Embedder("dinov2_vitb14"),
        "dinov2_s_reg": lambda: DINOv2Embedder("dinov2_vits14_reg"),
        "clip": lambda: CLIPEmbedder(),
    }

    all_results: dict[str, list[ModelResult]] = {}

    for model_key in args.models:
        if model_key not in model_builders:
            print(f"Unknown model: {model_key}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Loading {model_key}...")
        embedder = model_builders[model_key]()

        for scenario_name, (scenario, frames_dir) in scenarios.items():
            print(f"\n--- {scenario_name} ---")
            result = benchmark_model(embedder, scenario, frames_dir)
            print_model_result(result, scenario_name)

            if scenario_name not in all_results:
                all_results[scenario_name] = []
            all_results[scenario_name].append(result)

        embedder.close()

    print_comparison_table(all_results)

    # Model size estimates
    print(f"\n{'=' * 80}")
    print("MODEL SIZE ESTIMATES (for on-device deployment)")
    print("=" * 80)
    sizes = {
        "MobileNetV3-Large": ("10 MB", "TFLite (already deployed)", "~15ms GPU"),
        "DINOv2-S/14": ("~85 MB (FP32), ~22 MB (INT8)", "needs ONNX→TFLite", "~40-80ms GPU est."),
        "DINOv2-B/14": ("~340 MB (FP32), ~85 MB (INT8)", "needs ONNX→TFLite", "~100-200ms GPU est."),
        "DINOv2-S/14-reg": ("~85 MB (FP32)", "needs ONNX→TFLite", "~40-80ms GPU est."),
        "CLIP-ViT-B/32": ("~340 MB (FP32), ~85 MB (INT8)", "needs ONNX→TFLite", "~80-150ms GPU est."),
    }
    for name, (size, conversion, latency) in sizes.items():
        print(f"  {name:<20} {size:<30} {conversion:<25} {latency}")


if __name__ == "__main__":
    main()
