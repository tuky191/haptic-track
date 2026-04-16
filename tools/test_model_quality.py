"""Pytest suite that validates embedding model quality against annotated scenarios.

Discovers all scenario JSON files in test_fixtures/, runs the model, and asserts:
  1. Same-object similarity > different-object similarity (separation)
  2. All expected re-acquisition scenarios pass
  3. Minimum separation gap between same/different object embeddings

Run:
    cd tools && ../.venv/bin/pytest test_model_quality.py -v

To test a different model:
    MODEL_PATH=/path/to/model.tflite pytest test_model_quality.py -v
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmark_embeddings import create_embedder, crop_and_embed, cosine_similarity

TOOLS_DIR = Path(__file__).parent
FIXTURES_DIR = TOOLS_DIR / "test_fixtures"
DEFAULT_MODEL = str(
    TOOLS_DIR.parent / "app/src/main/assets/mobilenet_v3_large_embedder.tflite"
)

# Thresholds — tune these as we gather real data
MIN_SEPARATION_GAP = 0.05       # same-object mean minus different-object mean
MIN_SAME_OBJECT_SIM = 0.3       # floor for same-object pairs (MobileNetV3 Large, 1280-dim)
MAX_DIFFERENT_OBJECT_SIM = 0.95  # ceiling for different-object pairs


def get_model_path():
    return os.environ.get("MODEL_PATH", DEFAULT_MODEL)


def discover_scenarios():
    """Find all scenario JSON files in test_fixtures/."""
    if not FIXTURES_DIR.exists():
        return []
    scenarios = []
    for scenario_json in FIXTURES_DIR.glob("*/scenario.json"):
        scenarios.append(scenario_json)
    return scenarios


def load_scenario(scenario_path: Path):
    scenario = json.loads(scenario_path.read_text())
    frames_dir = scenario_path.parent / "frames"
    return scenario, frames_dir


scenarios = discover_scenarios()


@pytest.fixture(scope="session")
def embedder():
    model_path = get_model_path()
    emb = create_embedder(model_path)
    yield emb
    emb.close()


@pytest.fixture(scope="session")
def all_embeddings(embedder):
    """Pre-compute all embeddings across all scenarios."""
    results = {}
    for scenario_path in scenarios:
        scenario, frames_dir = load_scenario(scenario_path)
        key = scenario_path.parent.name
        results[key] = {}

        for frame_ann in scenario["frames"]:
            frame_path = frames_dir / frame_ann["frame"]
            if not frame_path.exists():
                continue
            for obj in frame_ann["objects"]:
                emb = crop_and_embed(embedder, frame_path, obj["box"])
                if emb is not None:
                    emb_key = f"{obj['object_id']}@{frame_ann['frame']}"
                    results[key][emb_key] = {
                        "object_id": obj["object_id"],
                        "frame": frame_ann["frame"],
                        "embedding": emb,
                    }
    return results


@pytest.mark.skipif(not scenarios, reason="No test scenarios in test_fixtures/")
class TestSameVsDifferent:
    """Verify the model can distinguish same vs different objects."""

    @pytest.mark.parametrize(
        "scenario_path", scenarios, ids=[s.parent.name for s in scenarios]
    )
    def test_separation_gap(self, scenario_path, all_embeddings):
        scenario, _ = load_scenario(scenario_path)
        key = scenario_path.parent.name
        embs = all_embeddings.get(key, {})

        if len(embs) < 2:
            pytest.skip("Not enough embeddings to compare")

        same_sims, diff_sims = [], []
        items = list(embs.values())

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sim = cosine_similarity(items[i]["embedding"], items[j]["embedding"])
                if items[i]["object_id"] == items[j]["object_id"]:
                    same_sims.append(sim)
                else:
                    diff_sims.append(sim)

        if not same_sims or not diff_sims:
            pytest.skip("Need both same-object and different-object pairs")

        gap = np.mean(same_sims) - np.mean(diff_sims)
        assert gap > MIN_SEPARATION_GAP, (
            f"Separation gap {gap:.3f} below threshold {MIN_SEPARATION_GAP}. "
            f"Same-object mean={np.mean(same_sims):.3f}, "
            f"different-object mean={np.mean(diff_sims):.3f}"
        )

    @pytest.mark.parametrize(
        "scenario_path", scenarios, ids=[s.parent.name for s in scenarios]
    )
    def test_same_object_floor(self, scenario_path, all_embeddings):
        scenario, _ = load_scenario(scenario_path)
        key = scenario_path.parent.name
        embs = all_embeddings.get(key, {})
        items = list(embs.values())

        same_sims = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i]["object_id"] == items[j]["object_id"]:
                    sim = cosine_similarity(items[i]["embedding"], items[j]["embedding"])
                    same_sims.append((sim, items[i], items[j]))

        if not same_sims:
            pytest.skip("No same-object pairs")

        for sim, a, b in same_sims:
            assert sim > MIN_SAME_OBJECT_SIM, (
                f"{a['object_id']}@{a['frame']} vs {b['frame']}: "
                f"similarity {sim:.3f} below floor {MIN_SAME_OBJECT_SIM}"
            )


@pytest.mark.skipif(not scenarios, reason="No test scenarios in test_fixtures/")
class TestReacquisition:
    """Verify expected re-acquisition outcomes from scenario definitions."""

    @pytest.mark.parametrize(
        "scenario_path", scenarios, ids=[s.parent.name for s in scenarios]
    )
    def test_reacquisition_scenarios(self, scenario_path, all_embeddings):
        scenario, _ = load_scenario(scenario_path)
        key = scenario_path.parent.name
        embs = all_embeddings.get(key, {})

        expected_list = scenario.get("expected_reacquisitions", [])
        if not expected_list:
            pytest.skip("No expected reacquisitions defined")

        for expected in expected_list:
            lock = expected["lock_on"]
            target = expected["should_reacquire"]
            distractors = expected.get("should_not_reacquire", [])

            lock_key = f"{lock['object_id']}@{lock['frame']}"
            target_key = f"{target['object_id']}@{target['frame']}"

            if lock_key not in embs or target_key not in embs:
                continue

            lock_emb = embs[lock_key]["embedding"]
            target_sim = cosine_similarity(lock_emb, embs[target_key]["embedding"])

            for dist in distractors:
                dist_key = f"{dist['object_id']}@{dist['frame']}"
                if dist_key not in embs:
                    continue
                dist_sim = cosine_similarity(lock_emb, embs[dist_key]["embedding"])
                assert target_sim > dist_sim, (
                    f"{expected['description']}: "
                    f"target {target['object_id']} sim={target_sim:.3f} should beat "
                    f"distractor {dist['object_id']} sim={dist_sim:.3f}"
                )
