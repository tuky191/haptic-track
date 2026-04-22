#!/usr/bin/env python3
"""Convert MobileViTv2-075 to TFLite for on-device embedding.

MobileViTv2's LinearSelfAttention uses broadcast mul ops that TFLite can't
handle directly. This script patches them with explicit expand operations
before converting via litert-torch.

Requirements:
    pip install torch timm litert-torch

Usage:
    cd tools && .venv/bin/python convert_mobilevitv2.py

Output: ../app/src/main/assets/mobilevitv2_075_embedder.tflite
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import timm.models.mobilevit as mvit


class TFLiteLinearSelfAttention(nn.Module):
    """TFLite-compatible LinearSelfAttention.

    Replaces implicit broadcast mul and expand_as with explicit expand calls
    that litert-torch can lower to TFLite ops.
    """

    def __init__(self, orig: mvit.LinearSelfAttention):
        super().__init__()
        self.embed_dim = orig.embed_dim
        self.qkv_proj = orig.qkv_proj
        self.attn_drop = orig.attn_drop
        self.out_proj = orig.out_proj
        self.out_drop = orig.out_drop

    def forward(self, x: torch.Tensor, x_prev=None) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)

        # Explicit expand instead of broadcast mul [B,1,P,N] * [B,d,P,N]
        context_scores_expanded = context_scores.expand(-1, self.embed_dim, -1, -1)
        context_vector = (key * context_scores_expanded).sum(dim=-1, keepdim=True)

        # Explicit expand instead of expand_as [B,d,P,1] -> [B,d,P,N]
        context_vector_expanded = context_vector.expand(-1, -1, -1, x.shape[-1])
        out = F.relu(value) * context_vector_expanded

        out = self.out_proj(out)
        out = self.out_drop(out)
        return out


def patch_model(model: nn.Module) -> int:
    """Replace all LinearSelfAttention modules with TFLite-compatible versions."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, mvit.LinearSelfAttention):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], TFLiteLinearSelfAttention(module))
            count += 1
    return count


def main():
    import litert_torch

    print("Loading MobileViTv2-075...")
    model = timm.create_model("mobilevitv2_075", pretrained=True, num_classes=0)
    model.eval()

    count = patch_model(model)
    print(f"Patched {count} LinearSelfAttention modules")

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output: {out.shape} ({out.shape[-1]}-dim embedding)")

    print("Converting to TFLite...")
    edge_model = litert_torch.convert(model, (dummy,))

    output_path = str(Path(__file__).parent.parent / "app/src/main/assets/mobilevitv2_075_embedder.tflite")
    edge_model.export(output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
