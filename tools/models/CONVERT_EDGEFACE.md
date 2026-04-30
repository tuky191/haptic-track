# EdgeFace-XS → TFLite recipe

Converts EdgeFace-XS (IJCB 2023) PyTorch weights to deployment-ready TFLite
for on-device face embedding. Replaces MobileFaceNet (PR #127).

**Shipped model**: `edgeface_xs_gamma_06` — EdgeNeXt backbone, 1.77M params,
512-dim embedding, 99.73% LFW.

## Step 1 — Export PyTorch → ONNX (macOS)

EdgeFace uses GELU (exact, via `erf`) which TFLite doesn't support natively.
Replace with the tanh approximation before export — cosine 0.99999 to exact.

```bash
.venv/bin/python3 - <<'PY'
import torch
import torch.nn as nn

model = torch.hub.load('otroshi/edgeface', 'edgeface_xs_gamma_06',
                        source='github', pretrained=True, trust_repo=True)
model.eval()

class GELUTanh(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x, approximate='tanh')

for name, module in model.named_modules():
    for child_name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, child_name, GELUTanh())

dummy = torch.randn(1, 3, 112, 112)
torch.onnx.export(model, dummy, 'tools/models/onnx/edgeface_xs_gamma_06.onnx',
                  input_names=['input'], output_names=['embedding'],
                  opset_version=17, dynamo=False)
PY
```

## Step 2 — ONNX → TFLite (Docker)

```bash
docker run --rm -v "$PWD/tools/models":/data python:3.10 \
  bash /data/convert_in_docker.sh edgeface_xs_gamma_06
```

> The `convert_in_docker.sh` pickle patch uses a hardcoded input shape. For
> EdgeFace, patch `np.random.rand(1, 3, 112, 112)` instead of `(1, 3, 256, 128)`.

Outputs in `tools/models/tflite/edgeface_xs_gamma_06/`:
- `edgeface_xs_gamma_06_float16.tflite` (3.8 MB) — **shipped**
- `edgeface_xs_gamma_06_float32.tflite` (7.3 MB) — reference

## Step 3 — Verify Python TFLite CPU

```bash
.venv/bin/python3 -c "
import tensorflow as tf, numpy as np
interp = tf.lite.Interpreter('tools/models/tflite/edgeface_xs_gamma_06/edgeface_xs_gamma_06_float16.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f'Input: {inp[\"shape\"]}  Output: {out[\"shape\"]}')
dummy = np.random.randn(*inp['shape']).astype(np.float32)
interp.set_tensor(inp['index'], dummy)
interp.invoke()
result = interp.get_tensor(out['index'])[0]
print(f'Dim: {result.shape}, norm: {np.linalg.norm(result):.4f}')
"
```

## Step 4 — Deploy and verify on-device

```bash
cp tools/models/tflite/edgeface_xs_gamma_06/edgeface_xs_gamma_06_float16.tflite \
   app/src/main/assets/edgeface_xs_gamma_06.tflite
```

Build, install, launch — `FaceEmbedder` logs:

```
FaceEmbed: Loaded EdgeFace-XS (512-dim) + BlazeFace
```

I/O contract:
- Input: `[1, 112, 112, 3]` NHWC float32, normalized `(pixel - 127.5) / 127.5`
- Output: `[1, 512]` float32, **L2-normalize before use**

## Key differences from MobileFaceNet

| | MobileFaceNet | EdgeFace-XS |
|---|---|---|
| Embedding dim | 192 | 512 |
| Batch size | Fixed 2 (slot 1 = zeros) | 1 |
| Preprocessing | `(p - 127.5) / 128.0` | `(p - 127.5) / 127.5` |
| Model size | 5.0 MB | 3.8 MB (FP16) |
| LFW | 99.55% | 99.73% |
| Architecture | CNN | Hybrid CNN+Transformer (EdgeNeXt) |
| TFLite ops | Standard | Standard (GELU replaced with tanh approx) |
