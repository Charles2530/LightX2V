"""Regression for int32 pointer overflow in H3 image-conditioned FFNs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_swiglu_rows_beyond_int32_element_offset():
    from lightx2v.common.ops.mm.fused_swiglu_linear import fused_swiglu_quant_fp8_f16_accum

    # Match the real I2AV [75419, 28672] FFN output. Only two tail rows
    # need values, but their pointer offsets must exceed signed int32.
    rows, width = 75419, 14336
    x = torch.empty((rows, 2 * width), device="cuda", dtype=torch.bfloat16)
    x[-2:, :width] = 2.0
    x[-2:, width:] = 1.0
    quantized, scales = fused_swiglu_quant_fp8_f16_accum(*x.chunk(2, dim=-1), 7.0)
    torch.cuda.synchronize()
    expected = (torch.tensor(2.0, device="cuda", dtype=x.dtype) * torch.nn.functional.silu(torch.tensor(1.0, device="cuda", dtype=x.dtype))).float()
    torch.testing.assert_close(scales[-2:], (expected / 7.0).expand(2), rtol=0, atol=0)
    torch.testing.assert_close(quantized[-2:].float(), torch.full((2, width), 7.0, device="cuda"), rtol=0, atol=0)


"""CUDA_VISIBLE_DEVICES=0 .venv_h3_runtime/bin/python -m pytest /mnt/afs_1/charles/codes/5090/LightX2V-main/tests/test_h3_swiglu_large_stride.py"""
