"""Pytest suite that gates model swaps via comparative benchmark.

Loads a benchmark results JSON (produced by benchmark_compare.py --output)
and asserts the candidate doesn't regress.

Usage:
    # First, run the benchmark:
    .venv/bin/python benchmark_compare.py \
        --incumbent mediapipe:../app/src/main/assets/mobilenet_v3_large_embedder.tflite \
        --candidate tflite:path/to/candidate.tflite \
        --output benchmark_results.json

    # Then, run the gate tests:
    BENCHMARK_RESULTS=benchmark_results.json .venv/bin/pytest test_benchmark_compare.py -v

    # Or, run both in one step (slower — re-runs inference):
    .venv/bin/pytest test_benchmark_compare.py -v \
        --incumbent mediapipe:../app/src/main/assets/mobilenet_v3_large_embedder.tflite \
        --candidate tflite:path/to/candidate.tflite
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).parent
FIXTURES_DIR = TOOLS_DIR / "test_fixtures"


@pytest.fixture(scope="session")
def benchmark_results(request):
    results_path = os.environ.get("BENCHMARK_RESULTS")
    if results_path:
        return json.loads(Path(results_path).read_text())

    inc_spec = request.config.getoption("--incumbent")
    cand_spec = request.config.getoption("--candidate")
    if not inc_spec or not cand_spec:
        pytest.skip(
            "Set BENCHMARK_RESULTS env var or pass --incumbent and --candidate"
        )

    from benchmark_compare import discover_scenarios, run_model, to_json

    scenarios = discover_scenarios(FIXTURES_DIR)
    if not scenarios:
        pytest.skip("No scenarios in test_fixtures/")

    inc = run_model(inc_spec, scenarios)
    cand = run_model(cand_spec, scenarios)
    return to_json(inc, cand)


class TestNoRegression:
    def test_verdict_not_worse(self, benchmark_results):
        assert benchmark_results["verdict"] != "WORSE", (
            f"Candidate regressed: gap delta = {benchmark_results['gap_delta']:+.3f}"
        )

    def test_gap_delta_positive_or_neutral(self, benchmark_results):
        assert benchmark_results["gap_delta"] >= -0.02, (
            f"Gap delta {benchmark_results['gap_delta']:+.3f} below -0.02 tolerance"
        )

    def test_no_reacquisition_regression(self, benchmark_results):
        inc = benchmark_results["incumbent"]
        cand = benchmark_results["candidate"]
        inc_pass = inc["total_reacq_pass"]
        cand_pass = cand["total_reacq_pass"]
        inc_total = inc_pass + inc["total_reacq_fail"]
        cand_total = cand_pass + cand["total_reacq_fail"]
        if inc_total == 0:
            pytest.skip("No reacquisition tests in scenarios")
        assert cand_pass >= inc_pass, (
            f"Reacquisition regressed: candidate {cand_pass}/{cand_total} "
            f"vs incumbent {inc_pass}/{inc_total}"
        )


class TestPerScenario:
    def test_no_scenario_gap_regression(self, benchmark_results):
        inc_scenarios = {
            s["name"]: s for s in benchmark_results["incumbent"]["scenarios"]
        }
        for cand_s in benchmark_results["candidate"]["scenarios"]:
            inc_s = inc_scenarios.get(cand_s["name"])
            if inc_s is None:
                continue
            if inc_s["same"]["n"] == 0 or inc_s["diff"]["n"] == 0:
                continue
            delta = cand_s["gap"] - inc_s["gap"]
            assert delta >= -0.05, (
                f"Scenario '{cand_s['name']}' gap regressed by {delta:+.3f} "
                f"(inc={inc_s['gap']:+.3f}, cand={cand_s['gap']:+.3f})"
            )
