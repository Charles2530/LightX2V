"""Opt-in equivalent patch projection for the Qwen vision conditioner."""

import json
from pathlib import Path


def install(output_dir):
    import torch
    import torch.nn.functional as F

    from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision import _PatchEmbed

    original = _PatchEmbed.forward
    checked = {"done": False}

    def forward(self, pixels):
        # Every input row is exactly one Conv3d receptive field; no patches
        # overlap. Flattening C,T,H,W therefore gives the same linear map.
        conv = self.proj
        assert conv.groups == 1 and tuple(conv.padding) == (0, 0, 0)
        assert tuple(conv.dilation) == (1, 1, 1) and conv.stride == conv.kernel_size
        flat = pixels.reshape(-1, self.in_channels * self.temporal_patch_size * self.patch_size**2).to(conv.weight.dtype)
        weight = conv.weight.flatten(1)
        if not checked["done"]:
            # Validate actual processor output before using the substitution.
            # This runs on the first image request (the warmup if requested).
            n = min(64, flat.shape[0])
            reference = original(self, pixels[:n])
            candidate = F.linear(flat[:n], weight, conv.bias)
            torch.cuda.synchronize()
            x, y = candidate.float(), reference.float()
            error = (torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y)).item()
            record = {
                "operation": "Conv3d whole-patch projection -> linear",
                "shape": list(flat.shape),
                "validation_patches": n,
                "dtype": str(flat.dtype),
                "relative_l2": error,
                "max_abs": (x - y).abs().max().item(),
                "finite": bool(torch.isfinite(x).all()),
                "original_weights_unchanged": True,
                "phase": "first vision request (warmup when requested)",
            }
            Path(output_dir, "vision_patch_validation.json").write_text(json.dumps(record, indent=2) + "\n")
            assert record["finite"] and error < 0.02, record
            checked["done"] = True
        return F.linear(flat, weight, conv.bias)

    _PatchEmbed.forward = forward
