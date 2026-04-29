#!/bin/bash
# Run inside the python:3.10 docker container with /data mapped to tools/models.
# Usage: bash /data/convert_in_docker.sh [model_name]
#   model_name defaults to osnet_ibn_x1_0_msmt17
#   Converts /data/onnx/<model_name>.onnx → /data/tflite/<model_name>/
set -e

MODEL_NAME="${1:-osnet_ibn_x1_0_msmt17}"
echo "=== converting $MODEL_NAME ==="

echo "=== installing deps ==="
# onnxsim 0.4.x+ requires cmake at install time on Python 3.10 — install it
# and pin onnxsim to a version known to build cleanly. Explicit numpy<2 because
# tensorflow + onnx2tf still expect the legacy numpy API.
apt-get update -qq && apt-get install -y -qq cmake >/dev/null
pip install --quiet "numpy<2" \
  tensorflow tf_keras "onnx<1.17" onnx2tf onnx_graphsurgeon psutil "onnxsim<0.5" sng4onnx ai-edge-litert 2>&1 | tail -3

CFPATH=/usr/local/lib/python3.10/site-packages/onnx2tf/utils/common_functions.py
echo "=== patching onnx2tf at $CFPATH ==="
python <<'PYEOF'
import re
path = "/usr/local/lib/python3.10/site-packages/onnx2tf/utils/common_functions.py"
with open(path) as f:
    code = f.read()
patched = re.sub(
    r"def download_test_image_data\(\).*?return test_image_data",
    'def download_test_image_data():\n    import numpy as np\n    test_image_data = np.random.rand(1, 3, 256, 128).astype(np.float32)\n    return test_image_data',
    code, flags=re.DOTALL,
)
with open(path, "w") as f:
    f.write(patched)
print("patched =>", "ok" if patched != code else "no-change")
PYEOF

cd /data
echo "=== running onnx2tf ==="
onnx2tf -i "onnx/${MODEL_NAME}.onnx" -o "tflite/${MODEL_NAME}" -cotof -n 2>&1 | tail -50

echo "=== output ==="
ls -la "tflite/${MODEL_NAME}/" | head -20
