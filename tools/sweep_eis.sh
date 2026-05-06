#!/bin/bash
# Quick parameter sweep for EIS bench — generates stabilized videos and measures quality.
# Usage: ./tools/sweep_eis.sh [BENCH_DIR] [RAW_VIDEO]

BENCH_DIR="${1:-bench_data/session_20260504_193511}"
RAW_VIDEO="${2:-bench_data/HapticTrack_20260504_193511_640.mp4}"
OUT_DIR="bench_data/sweep_results"
mkdir -p "$OUT_DIR"

source .venv/bin/activate

run_variant() {
    local name="$1"
    shift
    local out_video="$OUT_DIR/${name}.mp4"
    echo "=== $name ==="
    python tools/stabilization_bench.py "$BENCH_DIR" "$RAW_VIDEO" --output-video "$@" 2>&1 | grep -E "Gyro replay \(adaptive|fixed TC\)|Non-causal"
    cp "$BENCH_DIR/results/stabilized_gyro.mp4" "$out_video"
    python tools/eis_quality.py "$out_video" 2>&1 | grep -E "^\s+(stabilized|${name})" || python tools/eis_quality.py "$out_video" 2>&1 | grep "PC jitter"
    echo ""
}

echo "Measuring ISP baseline..."
python tools/eis_quality.py "$RAW_VIDEO" 2>&1 | grep "PC jitter"
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

# Crop zoom variations
run_variant "tc070_crop110" --tc 0.70 --crop 1.10
run_variant "tc070_crop120" --tc 0.70 --crop 1.20

# Gaussian kernel (non-causal, for video capture)
run_variant "gaussian_400" --tc 0.70 --gaussian 400
run_variant "gaussian_600" --tc 0.70 --gaussian 600

# Hybrid: gyro rotation + optical flow translation
run_variant "hybrid_tc015" --tc 0.70 --hybrid --translation-tc 0.15
run_variant "hybrid_tc030" --tc 0.70 --hybrid --translation-tc 0.30

echo "=== SUMMARY ==="
echo "Measuring all variants..."
python tools/eis_quality.py "$OUT_DIR"/*.mp4 --raw "$RAW_VIDEO" --output "$OUT_DIR"
