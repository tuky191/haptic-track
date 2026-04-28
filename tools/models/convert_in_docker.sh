#!/bin/bash
# Run inside the python:3.10 docker container with /data mapped to tools/models.
set -e

echo "=== installing deps ==="
pip install --quiet \
  tensorflow tf_keras "onnx<1.17" onnx2tf onnx_graphsurgeon psutil onnxsim sng4onnx ai-edge-litert 2>&1 | tail -3

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
onnx2tf -i onnx/osnet_ibn_x1_0_msmt17.onnx -o tflite/osnet_ibn_x1_0_msmt17 -cotof -n 2>&1 | tail -50

echo "=== output ==="
ls -la tflite/osnet_ibn_x1_0_msmt17/ | head -20
