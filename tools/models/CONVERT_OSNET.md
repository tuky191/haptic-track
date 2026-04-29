# OSNet-IBN x1.0 MSMT17 → TFLite recipe

Converts the torchreid OSNet-IBN x1.0 cross-domain weights from
[Hugging Face mirror kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet)
into a deployment-ready TFLite for on-device person re-identification.

This is the recommended swap target if SessionRoster (#108) leaves
residual identity confusion (see `tools/crops/baseline_results.md` for
gap measurements).

## Step 1 — Export PyTorch → ONNX (macOS)

Requires the root `.venv` with `torch`, `torchreid`, `onnx`, `onnxscript`,
`onnxruntime`. Run from `tools/`:

```bash
mkdir -p models/osnet_hf models/onnx models/tflite

# Pull pretrained weights
curl -L -o models/osnet_hf/osnet_ibn_x1_0_msmt17_combineall.pth \
  "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"

# Export PyTorch → ONNX
../.venv/bin/python - <<'PY'
import torch
from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights

model = build_model(name='osnet_ibn_x1_0', num_classes=1041, pretrained=False)
load_pretrained_weights(model, 'models/osnet_hf/osnet_ibn_x1_0_msmt17_combineall.pth')
model.eval()

dummy = torch.randn(1, 3, 256, 128)
torch.onnx.export(model, dummy, 'models/onnx/osnet_ibn_x1_0_msmt17.onnx',
                  input_names=['input'], output_names=['embedding'],
                  opset_version=13)
PY

# PyTorch 2.11+ may emit external-data ONNX. Combine into single file.
../.venv/bin/python - <<'PY'
import onnx
m = onnx.load('models/onnx/osnet_ibn_x1_0_msmt17.onnx', load_external_data=True)
onnx.save_model(m, 'models/onnx/osnet_ibn_x1_0_msmt17_single.onnx',
                save_as_external_data=False)
PY
mv models/onnx/osnet_ibn_x1_0_msmt17_single.onnx models/onnx/osnet_ibn_x1_0_msmt17.onnx
rm -f models/onnx/osnet_ibn_x1_0_msmt17.onnx.data
```

## Step 2 — ONNX → TFLite (Docker, macOS TF crash workaround)

`onnx2tf` crashes on macOS ARM64. Run inside a Linux Docker container:

```bash
docker run --rm -v "$PWD/models":/data python:3.10 bash /data/convert_in_docker.sh
```

The mounted script:
1. Installs tensorflow + onnx2tf + supporting libs
2. Patches `onnx2tf.utils.common_functions.download_test_image_data` to
   bypass a broken numpy pickle download (test data path doesn't exist;
   we substitute random tensor data of the right shape)
3. Runs `onnx2tf -i osnet_ibn_x1_0_msmt17.onnx -o osnet_ibn_x1_0_msmt17 -cotof -n`

Outputs in `models/tflite/osnet_ibn_x1_0_msmt17/`:
- `osnet_ibn_x1_0_msmt17_float16.tflite` (4.5 MB) — recommended for deployment
- `osnet_ibn_x1_0_msmt17_float32.tflite` (8.7 MB) — full precision reference

## Step 3 — Verify round-trip

```bash
../.venv/bin/python - <<'PY'
import numpy as np, tensorflow as tf, torch
from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights

model = build_model(name='osnet_ibn_x1_0', num_classes=1041, pretrained=False)
load_pretrained_weights(model, 'models/osnet_hf/osnet_ibn_x1_0_msmt17_combineall.pth')
model.eval()

torch.manual_seed(42); np.random.seed(42)
x = torch.randn(1, 3, 256, 128)
with torch.no_grad():
    y_torch = model(x).numpy()
y_torch /= np.linalg.norm(y_torch)

for variant in ['float32', 'float16']:
    interp = tf.lite.Interpreter(
        model_path=f'models/tflite/osnet_ibn_x1_0_msmt17/osnet_ibn_x1_0_msmt17_{variant}.tflite')
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    arr = np.transpose(x.numpy(), (0, 2, 3, 1)).astype(inp['dtype'])
    interp.set_tensor(inp['index'], arr); interp.invoke()
    y_tflite = interp.get_tensor(out['index']).reshape(1, -1)
    y_tflite /= np.linalg.norm(y_tflite)
    cos = float(np.dot(y_torch.flatten(), y_tflite.flatten()))
    print(f'{variant}: cosine={cos:.5f}')
PY
```

Expected:
- FP32: cosine 1.00000
- FP16: cosine 0.99996

## Step 4 — Deploy

Copy the FP16 TFLite into `app/src/main/assets/` and update
`PersonReIdEmbedder.kt` if the deployment is approved (after
SessionRoster #108 lands and residual is measured):

```bash
cp models/tflite/osnet_ibn_x1_0_msmt17/osnet_ibn_x1_0_msmt17_float16.tflite \
   ../app/src/main/assets/osnet_ibn_x1_0_msmt17.tflite
```

Update `PersonReIdEmbedder.kt`:
- `private const val MODEL_ASSET = "osnet_ibn_x1_0_msmt17.tflite"`

Both models share the same I/O contract:
- Input: `[1, 256, 128, 3]` NHWC float32, ImageNet-normalized
  (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Output: `[1, 512]` float32, **L2-normalize before use**
- Same `CanonicalCropper` letterbox preprocessing — no plumbing change

After swap, re-run all replay tests and consider re-tuning
`PERSON_REID_FLOOR` (currently 0.45) — different model means different
distribution. Off-device benchmark numbers in
`tools/crops/baseline_results.md` suggest the OSNet-IBN diff_p90 is
~0.55 (vs 0.77 for current Market-trained), so the floor likely needs
to come DOWN to maintain rejection sensitivity.
