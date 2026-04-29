# OSNet → TFLite recipes

Converts torchreid OSNet weights from
[Hugging Face mirror kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet)
into deployment-ready TFLite for on-device person re-identification.

This file documents two variants:

- **`osnet_x1_0_msmt17`** — the **currently shipped model**. Base x1_0
  architecture (no IBN/AIN), trained on MSMT17. Drop-in upgrade over the
  original `osnet_x1_0_market` — same architecture (so Adreno GPU delegate
  doesn't trip over instance-norm), better training distribution.
- **`osnet_ibn_x1_0_msmt17`** — investigated in #117 and **not** shipped.
  The Adreno GPU delegate silently miscompiles the IBN (Instance/Batch
  Normalization) branch of OSNet-IBN, producing a near-constant output
  vector regardless of input. Same TFLite file works correctly on Python
  CPU and Android TFLite CPU; only the Adreno GPU delegate breaks. The
  `PersonReIdEmbedder` self-test catches this at init via the bundled
  `assets/reid_selftest_{a,b,c}.jpg` distinct-scene crops — if max pairwise
  cosine > 0.99, the embedder auto-disables.

> **Before swapping in any new TFLite,** run `tools/test_osnet_tflite_cpu.py`
> to confirm Python CPU output is varied/discriminative, then deploy and
> watch logcat for `OSNet self-test passed`. The self-test catches silent
> GPU-delegate regressions that conversion-time validation cannot.

---

## A. Convert `osnet_x1_0_msmt17` (shipped)

### Step 1 — Export PyTorch → ONNX (macOS)

Requires the root `.venv` with `torch`, `torchreid`, `onnx`, `onnxscript`,
`onnxruntime`. Run from the repo root:

```bash
mkdir -p tools/models/osnet_hf tools/models/onnx tools/models/tflite

# Pull pretrained weights
curl -L -o tools/models/osnet_hf/osnet_x1_0_msmt17_combineall.pth \
  "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"

# Export PyTorch → ONNX
.venv/bin/python - <<'PY'
import torch
from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights

model = build_model(name='osnet_x1_0', num_classes=1041, pretrained=False)
load_pretrained_weights(model, 'tools/models/osnet_hf/osnet_x1_0_msmt17_combineall.pth')
model.eval()

dummy = torch.randn(1, 3, 256, 128)
torch.onnx.export(model, dummy, 'tools/models/onnx/osnet_x1_0_msmt17.onnx',
                  input_names=['input'], output_names=['embedding'],
                  opset_version=13)

# PyTorch 2.11+ may emit external-data ONNX. Combine into single file.
import onnx
m = onnx.load('tools/models/onnx/osnet_x1_0_msmt17.onnx', load_external_data=True)
onnx.save_model(m, 'tools/models/onnx/osnet_x1_0_msmt17_single.onnx',
                save_as_external_data=False)
PY
mv tools/models/onnx/osnet_x1_0_msmt17_single.onnx tools/models/onnx/osnet_x1_0_msmt17.onnx
rm -f tools/models/onnx/osnet_x1_0_msmt17.onnx.data
```

### Step 2 — ONNX → TFLite (Docker, macOS TF crash workaround)

`onnx2tf` crashes on macOS ARM64. Run inside a Linux Docker container:

```bash
docker run --rm -v "$PWD/tools/models":/data python:3.10 \
  bash /data/convert_in_docker.sh osnet_x1_0_msmt17
```

`convert_in_docker.sh` accepts the model name as an argument; it converts
`/data/onnx/<model>.onnx` → `/data/tflite/<model>/<model>_{float16,float32}.tflite`.

Outputs in `tools/models/tflite/osnet_x1_0_msmt17/`:

- `osnet_x1_0_msmt17_float16.tflite` (4.4 MB) — **shipped in `app/src/main/assets/`**
- `osnet_x1_0_msmt17_float32.tflite` (8.7 MB) — full-precision reference

### Step 3 — Verify Python TFLite CPU

```bash
tools/.venv/bin/python tools/test_osnet_tflite_cpu.py
```

The script feeds 6 person crops through each model and prints the pairwise
cosine matrix. A working model produces varied outputs and pairwise cosines
clustered around 0.93-0.99 for same-person crops. A degenerate model (e.g.
the broken IBN) produces near-identical output on different inputs.

### Step 4 — Deploy and verify on-device

```bash
cp tools/models/tflite/osnet_x1_0_msmt17/osnet_x1_0_msmt17_float16.tflite \
   app/src/main/assets/osnet_x1_0_msmt17.tflite
```

Update `PersonReIdEmbedder.kt`:

```kotlin
private const val MODEL_ASSET = "osnet_x1_0_msmt17.tflite"
```

Build, install, launch — `PersonReIdEmbedder` runs the self-test at init
and logs:

```
PersonReId: Loaded OSNet (512-dim, 256x128)
PersonReId: OSNet self-test passed: max pairwise cosine = 0.560896 (< 0.99)
```

If max pairwise cosine > 0.99, the self-test logs an `E`-level error and
**auto-disables the embedder** (`embed()` returns null, cascade falls back
to MNV3-only scoring). That's the regression-catch path for any future
GPU-delegate miscompilation.

Both models share the same I/O contract:

- Input: `[1, 256, 128, 3]` NHWC float32, ImageNet-normalized
  (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Output: `[1, 512]` float32, **L2-normalize before use**
- Same `CanonicalCropper` letterbox preprocessing — no plumbing change

---

## B. Convert `osnet_ibn_x1_0_msmt17` (NOT shipped — see #117)

The IBN variant has been investigated and **does not work on Adreno GPU
delegate** despite producing correct CPU output. The recipe is preserved
here so a future investigation (e.g. with a different GPU delegate, or
after an upstream TFLite fix) can rebuild it.

Same flow as section A but with `osnet_ibn_x1_0`:

```bash
curl -L -o tools/models/osnet_hf/osnet_ibn_x1_0_msmt17_combineall.pth \
  "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"

# Replace 'osnet_x1_0' → 'osnet_ibn_x1_0' in build_model() above

docker run --rm -v "$PWD/tools/models":/data python:3.10 \
  bash /data/convert_in_docker.sh osnet_ibn_x1_0_msmt17
```

Validate by deploying and confirming the self-test FAILS as expected
(max pairwise cosine ≈ 1.0). If a fix lands and the self-test passes on
Adreno GPU, the IBN variant becomes a candidate again — its off-device
benchmark numbers (`tools/crops/baseline_results.md`) suggest tighter
diff-person separation than the base x1_0.
