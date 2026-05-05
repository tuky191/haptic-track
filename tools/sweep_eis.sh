#!/bin/bash
# Quick parameter sweep for EIS bench — generates stabilized videos and measures quality.
# Usage: ./tools/sweep_eis.sh

BENCH_DIR="bench_data/session_20260504_193511"
RAW_VIDEO="bench_data/HapticTrack_20260504_193511_640.mp4"
OUT_DIR="bench_data/sweep_results"
mkdir -p "$OUT_DIR"

source .venv/bin/activate

run_variant() {
    local name="$1"
    shift
    local out_video="$OUT_DIR/${name}.mp4"
    echo "=== $name ==="
    python tools/stabilization_bench.py "$BENCH_DIR" "$RAW_VIDEO" --output-video "$@" 2>&1 | grep -E "Gyro replay \(adaptive|fixed TC\)|Non-causal"
    cp "$BENCH_DIR/results/stabilized.mp4" "$out_video"
    python tools/eis_quality.py "$out_video" 2>&1 | grep -E "^\s+(stabilized|${name})" || python tools/eis_quality.py "$out_video" 2>&1 | grep "PC jitter"
    echo ""
}

echo "Measuring ISP baseline..."
python tools/eis_quality.py bench_data/HapticTrack_20260504_193511_640.mp4 2>&1 | grep "PC jitter"
echo ""

# Current best: tc=0.70, adaptive, leash, ois=0.80
run_variant "tc070_adaptive" --tc 0.70

# Higher TC: more smoothing
run_variant "tc080_adaptive" --tc 0.80
run_variant "tc090_adaptive" --tc 0.90
run_variant "tc100_adaptive" --tc 1.00

# tc=0.70 without leash
run_variant "tc070_noleash" --tc 0.70 --no-leash

# tc=0.70 without adaptive
run_variant "tc070_fixed" --tc 0.70 --no-adaptive

# OIS compensation variations at tc=0.70
run_variant "tc070_ois100" --tc 0.70 --ois 1.0
run_variant "tc070_ois060" --tc 0.70 --ois 0.60

# Lower crop zoom (less amplification)
run_variant "tc070_crop110" --tc 0.70 --crop 1.10
run_variant "tc070_crop120" --tc 0.70 --crop 1.20

echo "=== SUMMARY ==="
echo "Measuring all variants..."
python tools/eis_quality.py "$OUT_DIR"/*.mp4 --raw "$RAW_VIDEO" --output "$OUT_DIR"
