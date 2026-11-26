"""The Qwen3-VL vision patch Conv3d and its Linear form must agree."""

import pytest

torch = pytest.importorskip("torch")


def test_patch_embed_linear_matches_conv3d():
    from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision import _PatchEmbed

    config = {"in_channels": 3, "temporal_patch_size": 2, "patch_size": 16, "hidden_size": 64}
    torch.manual_seed(0)
    conv = _PatchEmbed(config).double()
    linear = _PatchEmbed(config, use_linear=True).double()
    linear.load_state_dict(conv.state_dict())
    # The processor hands the tower flattened [patches, C*T*P*P] rows.
    pixels = torch.randn(37, 3 * 2 * 16 * 16, dtype=torch.float64)
    torch.testing.assert_close(linear(pixels), conv(pixels), rtol=1e-10, atol=1e-10)
