#!/usr/bin/env python3
"""Compare two embedding models on the annotated scenario corpus.

Runs incumbent and candidate models on the same crops, computes
same-vs-different cosine distributions, and produces a clear
"better / worse / same" verdict that isn't threshold-tunable.

Metrics:
  - gap (same_p10 - diff_p90): can a fixed threshold separate identities?
  - d-prime: signal detection sensitivity, pooled across scenarios
  - per-scenario and per-reacquisition-test pass/fail

Usage:
    cd tools && .venv/bin/python benchmark_compare.py \
        --incumbent mediapipe:../app/src/main/assets/mobilenet_v3_large_embedder.tflite \
        --candidate tflite:../path/to/candidate.tflite \
        [--scenarios test_fixtures] \
        [--output results.json]

Model spec formats:
    mediapipe:<path>     — MediaPipe ImageEmbedder (auto preprocessing)
    tflite:<path>        — TFLite direct interpreter (ImageNet normalization)
    onnx:<path>          — ONNX Runtime (ImageNet normalization)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


# ----------------------------------------------------------------- types

EmbedFn = Callable[[Image.Image], np.ndarray]


@dataclass
class DistributionStats:
    n: int
    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    min: float
    max: float

    @staticmethod
    def from_values(values: list[float]) -> DistributionStats:
        if not values:
            return DistributionStats(0, 0, 0, 0, 0, 0, 0, 0)
        a = np.array(values)
        return DistributionStats(
            n=len(a),
            mean=float(np.mean(a)),
            std=float(np.std(a)),
            p10=float(np.percentile(a, 10)),
            p50=float(np.percentile(a, 50)),
            p90=float(np.percentile(a, 90)),
            min=float(np.min(a)),
            max=float(np.max(a)),
        )


@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    n_embeddings: int
    same: DistributionStats
    diff: DistributionStats
    gap: float  # same_p10 - diff_p90
    dprime: float
    reacq_pass: int = 0
    reacq_fail: int = 0
    reacq_details: list[dict] = field(default_factory=list)


@dataclass
class ModelResult:
    model_spec: str
    scenarios: list[ScenarioResult] = field(default_factory=list)

    @property
    def aggregate_gap(self) -> float:
        gaps = [s.gap for s in self.scenarios if s.same.n > 0 and s.diff.n > 0]
        return float(np.mean(gaps)) if gaps else 0.0

    @property
    def aggregate_dprime(self) -> float:
        ds = [s.dprime for s in self.scenarios if s.same.n > 0 and s.diff.n > 0]
        return float(np.mean(ds)) if ds else 0.0

    @property
    def total_reacq_pass(self) -> int:
        return sum(s.reacq_pass for s in self.scenarios)

    @property
    def total_reacq_fail(self) -> int:
        return sum(s.reacq_fail for s in self.scenarios)


# ----------------------------------------------------------------- model backends


def _crop_box(img: Image.Image, box: list[float]) -> Image.Image:
    w, h = img.size
    left = int(box[0] * w)
    top = int(box[1] * h)
    right = int(box[2] * w)
    bottom = int(box[3] * h)
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    return img.crop((left, top, right, bottom))


def _resize_letterbox(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def build_mediapipe_embedder(model_path: str) -> EmbedFn:
    import mediapipe as mp

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=True, quantize=False,
    )
    embedder = mp.tasks.vision.ImageEmbedder.create_from_options(options)

    def embed(crop: Image.Image) -> np.ndarray:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.array(crop.convert("RGB")),
        )
        result = embedder.embed(mp_image)
        return np.array(result.embeddings[0].embedding, dtype=np.float32)

    embed._close = embedder.close
    return embed


def build_tflite_embedder(model_path: str) -> EmbedFn:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    shape = inp["shape"]
    nhwc = shape[-1] == 3
    h_idx, w_idx = (1, 2) if nhwc else (2, 3)
    h, w = int(shape[h_idx]), int(shape[w_idx])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def embed(crop: Image.Image) -> np.ndarray:
        img = _resize_letterbox(crop.convert("RGB"), w, h)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        if nhwc:
            arr = arr[None, ...]
        else:
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
        interpreter.set_tensor(inp["index"], arr.astype(np.float32))
        interpreter.invoke()
        feat = interpreter.get_tensor(out["index"]).reshape(-1).astype(np.float32)
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        return feat

    return embed


def build_onnx_embedder(model_path: str) -> EmbedFn:
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    nhwc = shape[-1] == 3
    h_idx, w_idx = (1, 2) if nhwc else (2, 3)
    h, w = int(shape[h_idx]), int(shape[w_idx])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def embed(crop: Image.Image) -> np.ndarray:
        img = _resize_letterbox(crop.convert("RGB"), w, h)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        if nhwc:
            arr = arr[None, ...]
        else:
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
        out = sess.run(None, {inp.name: arr.astype(np.float32)})
        feat = out[0].reshape(-1).astype(np.float32)
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        return feat

    return embed


def build_embedder(spec: str) -> EmbedFn:
    if ":" not in spec:
        raise ValueError(f"Model spec must be 'backend:path', got '{spec}'")
    backend, path = spec.split(":", 1)
    path = str(Path(path).resolve())
    if backend == "mediapipe":
        return build_mediapipe_embedder(path)
    elif backend == "tflite":
        return build_tflite_embedder(path)
    elif backend == "onnx":
        return build_onnx_embedder(path)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use mediapipe, tflite, or onnx.")


# ----------------------------------------------------------------- metrics


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-8 else 0.0


def d_prime(same_sims: list[float], diff_sims: list[float]) -> float:
    if not same_sims or not diff_sims:
        return 0.0
    mu_s, mu_d = np.mean(same_sims), np.mean(diff_sims)
    var_s, var_d = np.var(same_sims), np.var(diff_sims)
    pooled = 0.5 * (var_s + var_d)
    if pooled < 1e-12:
        return 0.0
    return float((mu_s - mu_d) / np.sqrt(pooled))


# ----------------------------------------------------------------- benchmark


def discover_scenarios(fixtures_dir: Path) -> list[Path]:
    if not fixtures_dir.exists():
        return []
    return sorted(fixtures_dir.glob("*/scenario.json"))


def run_scenario(
    embed_fn: EmbedFn,
    scenario_path: Path,
) -> ScenarioResult | None:
    scenario = json.loads(scenario_path.read_text())
    frames_dir = scenario_path.parent / "frames"
    name = scenario_path.parent.name

    if not frames_dir.exists():
        return None

    embeddings: dict[str, tuple[str, np.ndarray]] = {}

    for frame_ann in scenario["frames"]:
        frame_path = frames_dir / frame_ann["frame"]
        if not frame_path.exists():
            continue
        img = Image.open(frame_path).convert("RGB")
        for obj in frame_ann["objects"]:
            crop = _crop_box(img, obj["box"])
            if crop.width < 1 or crop.height < 1:
                continue
            emb = embed_fn(crop)
            key = f"{obj['object_id']}@{frame_ann['frame']}"
            embeddings[key] = (obj["object_id"], emb)

    if len(embeddings) < 2:
        return None

    same_sims: list[float] = []
    diff_sims: list[float] = []
    items = list(embeddings.values())

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = cosine_similarity(items[i][1], items[j][1])
            if items[i][0] == items[j][0]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    same_stats = DistributionStats.from_values(same_sims)
    diff_stats = DistributionStats.from_values(diff_sims)
    gap = same_stats.p10 - diff_stats.p90 if same_sims and diff_sims else 0.0
    dp = d_prime(same_sims, diff_sims)

    reacq_pass = 0
    reacq_fail = 0
    reacq_details = []

    for expected in scenario.get("expected_reacquisitions", []):
        lock = expected["lock_on"]
        target = expected["should_reacquire"]
        distractors = expected.get("should_not_reacquire", [])

        lock_key = f"{lock['object_id']}@{lock['frame']}"
        target_key = f"{target['object_id']}@{target['frame']}"

        if lock_key not in embeddings or target_key not in embeddings:
            continue

        lock_emb = embeddings[lock_key][1]
        target_sim = cosine_similarity(lock_emb, embeddings[target_key][1])

        passed = True
        detail = {
            "description": expected["description"],
            "target_sim": round(target_sim, 4),
            "distractors": {},
        }

        for dist in distractors:
            dist_key = f"{dist['object_id']}@{dist['frame']}"
            if dist_key not in embeddings:
                continue
            dist_sim = cosine_similarity(lock_emb, embeddings[dist_key][1])
            detail["distractors"][dist["object_id"]] = round(dist_sim, 4)
            if dist_sim >= target_sim:
                passed = False

        detail["passed"] = passed
        reacq_details.append(detail)
        if passed:
            reacq_pass += 1
        else:
            reacq_fail += 1

    return ScenarioResult(
        scenario_name=name,
        description=scenario.get("description", name),
        n_embeddings=len(embeddings),
        same=same_stats,
        diff=diff_stats,
        gap=gap,
        dprime=dp,
        reacq_pass=reacq_pass,
        reacq_fail=reacq_fail,
        reacq_details=reacq_details,
    )


def run_model(
    model_spec: str,
    scenario_paths: list[Path],
) -> ModelResult:
    embed_fn = build_embedder(model_spec)
    result = ModelResult(model_spec=model_spec)

    for sp in scenario_paths:
        sr = run_scenario(embed_fn, sp)
        if sr is not None:
            result.scenarios.append(sr)

    if hasattr(embed_fn, "_close"):
        embed_fn._close()

    return result


# ----------------------------------------------------------------- output


def print_comparison(incumbent: ModelResult, candidate: ModelResult):
    inc_name = Path(incumbent.model_spec.split(":", 1)[1]).stem
    cand_name = Path(candidate.model_spec.split(":", 1)[1]).stem

    print()
    print("=" * 78)
    print(f"  INCUMBENT: {inc_name}")
    print(f"  CANDIDATE: {cand_name}")
    print("=" * 78)

    all_scenarios = sorted(
        set(s.scenario_name for s in incumbent.scenarios)
        | set(s.scenario_name for s in candidate.scenarios)
    )

    inc_by_name = {s.scenario_name: s for s in incumbent.scenarios}
    cand_by_name = {s.scenario_name: s for s in candidate.scenarios}

    print()
    print(f"  {'Scenario':<25s}  {'Model':<20s}  {'Gap':>7s}  {'d-prime':>7s}  "
          f"{'Same(p10)':>9s}  {'Diff(p90)':>9s}  {'Reacq':>6s}")
    print("  " + "-" * 74)

    for sname in all_scenarios:
        inc_s = inc_by_name.get(sname)
        cand_s = cand_by_name.get(sname)

        for label, s in [("incumbent", inc_s), ("candidate", cand_s)]:
            if s is None:
                continue
            reacq_str = f"{s.reacq_pass}/{s.reacq_pass + s.reacq_fail}" if s.reacq_pass + s.reacq_fail > 0 else "n/a"
            same_p10 = f"{s.same.p10:+.3f}" if s.same.n > 0 else "n/a"
            diff_p90 = f"{s.diff.p90:+.3f}" if s.diff.n > 0 else "n/a"
            name_col = sname if label == "incumbent" else ""
            print(f"  {name_col:<25s}  {label:<20s}  {s.gap:+7.3f}  {s.dprime:7.2f}  "
                  f"{same_p10:>9s}  {diff_p90:>9s}  {reacq_str:>6s}")

        # Verdict per scenario
        if inc_s and cand_s and inc_s.same.n > 0 and cand_s.same.n > 0:
            delta = cand_s.gap - inc_s.gap
            if delta > 0.02:
                v = "BETTER"
            elif delta < -0.02:
                v = "WORSE"
            else:
                v = "SAME"
            print(f"  {'':25s}  {'→ ' + v:<20s}  {delta:+7.3f}")
        print()

    # Aggregate
    print("  " + "=" * 74)
    print(f"  {'AGGREGATE':<25s}  {'incumbent':<20s}  {incumbent.aggregate_gap:+7.3f}  "
          f"{incumbent.aggregate_dprime:7.2f}  {'':9s}  {'':9s}  "
          f"{incumbent.total_reacq_pass}/{incumbent.total_reacq_pass + incumbent.total_reacq_fail}")
    print(f"  {'':25s}  {'candidate':<20s}  {candidate.aggregate_gap:+7.3f}  "
          f"{candidate.aggregate_dprime:7.2f}  {'':9s}  {'':9s}  "
          f"{candidate.total_reacq_pass}/{candidate.total_reacq_pass + candidate.total_reacq_fail}")

    agg_delta = candidate.aggregate_gap - incumbent.aggregate_gap
    if agg_delta > 0.02:
        verdict = "BETTER"
    elif agg_delta < -0.02:
        verdict = "WORSE"
    else:
        verdict = "SAME"

    print()
    print(f"  VERDICT: {verdict} (gap delta = {agg_delta:+.3f})")
    print()

    # Reacquisition details
    any_details = any(s.reacq_details for s in candidate.scenarios)
    if any_details:
        print("  REACQUISITION DETAILS (candidate)")
        print("  " + "-" * 74)
        for s in candidate.scenarios:
            for d in s.reacq_details:
                status = "PASS" if d["passed"] else "FAIL"
                dists = ", ".join(f"{k}={v:.3f}" for k, v in d["distractors"].items())
                print(f"  [{status}] {s.scenario_name}: {d['description']}")
                print(f"         target={d['target_sim']:.3f}  {dists}")
        print()


def to_json(incumbent: ModelResult, candidate: ModelResult) -> dict:
    def stats_dict(s: DistributionStats) -> dict:
        return {"n": s.n, "mean": s.mean, "std": s.std,
                "p10": s.p10, "p50": s.p50, "p90": s.p90, "min": s.min, "max": s.max}

    def model_dict(m: ModelResult) -> dict:
        return {
            "model_spec": m.model_spec,
            "aggregate_gap": m.aggregate_gap,
            "aggregate_dprime": m.aggregate_dprime,
            "total_reacq_pass": m.total_reacq_pass,
            "total_reacq_fail": m.total_reacq_fail,
            "scenarios": [
                {
                    "name": s.scenario_name,
                    "description": s.description,
                    "n_embeddings": s.n_embeddings,
                    "same": stats_dict(s.same),
                    "diff": stats_dict(s.diff),
                    "gap": s.gap,
                    "dprime": s.dprime,
                    "reacq_pass": s.reacq_pass,
                    "reacq_fail": s.reacq_fail,
                    "reacq_details": s.reacq_details,
                }
                for s in m.scenarios
            ],
        }

    agg_delta = candidate.aggregate_gap - incumbent.aggregate_gap
    if agg_delta > 0.02:
        verdict = "BETTER"
    elif agg_delta < -0.02:
        verdict = "WORSE"
    else:
        verdict = "SAME"

    return {
        "verdict": verdict,
        "gap_delta": round(agg_delta, 4),
        "incumbent": model_dict(incumbent),
        "candidate": model_dict(candidate),
    }


# ----------------------------------------------------------------- CLI


def main():
    parser = argparse.ArgumentParser(
        description="Compare two embedding models on the scenario corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--incumbent", required=True,
        help="Incumbent model spec (e.g. mediapipe:path/to/model.tflite)",
    )
    parser.add_argument(
        "--candidate", required=True,
        help="Candidate model spec (e.g. tflite:path/to/candidate.tflite)",
    )
    parser.add_argument(
        "--scenarios", default="test_fixtures",
        help="Directory containing scenario subdirectories (default: test_fixtures)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write JSON results to this file",
    )
    args = parser.parse_args()

    scenario_paths = discover_scenarios(Path(args.scenarios))
    if not scenario_paths:
        print(f"No scenarios found in {args.scenarios}/*/scenario.json", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(scenario_paths)} scenarios: "
          f"{', '.join(p.parent.name for p in scenario_paths)}")

    print(f"\nRunning incumbent: {args.incumbent}")
    inc = run_model(args.incumbent, scenario_paths)

    print(f"Running candidate: {args.candidate}")
    cand = run_model(args.candidate, scenario_paths)

    print_comparison(inc, cand)

    if args.output:
        out = to_json(inc, cand)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
