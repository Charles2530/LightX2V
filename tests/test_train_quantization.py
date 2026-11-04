import unittest
import socket

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lightx2v_train.utils.quantization import NVFP4FakeQuantizer, QATLinear, normalize_quant_config


NVFP4 = {"bit": "nvfp4", "symmetric": True, "granularity": "per_group", "group_size": 16}


def _fsdp_worker(rank, port):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        linear = nn.Linear(16, 32, device=f"cuda:{rank}")
        qat = QATLinear(linear, normalize_quant_config({"weight": NVFP4, "act": NVFP4, "grad_output": NVFP4}))
        fully_shard(qat, mesh=init_device_mesh("cuda", (2,)), mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
        x = torch.randn(2, 3, 16, device=f"cuda:{rank}", dtype=torch.bfloat16, requires_grad=True)
        qat(x).square().mean().backward()
        grad_weight = qat.weight.grad
        if hasattr(grad_weight, "to_local"):
            grad_weight = grad_weight.to_local()
        assert torch.isfinite(x.grad).all() and torch.isfinite(grad_weight).all()
    finally:
        dist.destroy_process_group()


class TestQATLinearBackward(unittest.TestCase):
    def test_linear_gradients(self):
        for device, dtype in (("cpu", torch.float32), ("cuda", torch.bfloat16)):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            for quantize_grad in (False, True):
                for checkpointed in (False, True):
                    with self.subTest(device=device, quantize_grad=quantize_grad, checkpointed=checkpointed):
                        torch.manual_seed(42)
                        config = {"weight": NVFP4, "act": NVFP4}
                        if quantize_grad:
                            config["grad_output"] = NVFP4
                        linear = nn.Linear(16, 32, device=device, dtype=dtype)
                        qat = QATLinear(linear, normalize_quant_config(config))
                        x = torch.randn(2, 3, 16, device=device, dtype=dtype, requires_grad=True)
                        dy = torch.randn(2, 3, 32, device=device, dtype=dtype)
                        output = checkpoint(qat, x, use_reentrant=False) if checkpointed else qat(x)
                        (output * dy).sum().backward()

                        quantizer = NVFP4FakeQuantizer(NVFP4)
                        xq = quantizer.quantize(x.detach().float()).to(dtype)
                        wq = quantizer.quantize(linear.weight.detach().float()).to(dtype)
                        if quantize_grad:
                            dy_dx = quantizer.quantize(dy.reshape(-1, 32).float()).to(dtype)
                            w_dx = quantizer.quantize(wq.T.float()).to(dtype)
                            expected_dx = (dy_dx @ w_dx.T).reshape_as(x)
                            dy_dw = quantizer.quantize(F.pad(dy.reshape(-1, 32).T.float(), (0, 10))).to(dtype)
                            x_dw = quantizer.quantize(F.pad(xq.reshape(-1, 16).T.float(), (0, 10))).to(dtype)
                            expected_dw = dy_dw @ x_dw.T
                        else:
                            expected_dx = dy @ wq
                            expected_dw = dy.reshape(-1, 32).T @ xq.reshape(-1, 16)
                        expected_db = dy.sum(dim=(0, 1))
                        tolerance = 0.05 if dtype == torch.bfloat16 else 1e-5
                        torch.testing.assert_close(x.grad, expected_dx, rtol=tolerance, atol=tolerance)
                        torch.testing.assert_close(linear.weight.grad, expected_dw, rtol=tolerance, atol=tolerance)
                        torch.testing.assert_close(linear.bias.grad, expected_db, rtol=tolerance, atol=tolerance)

    def test_grad_output_config_validation(self):
        config = {"weight": NVFP4, "act": NVFP4}
        self.assertNotIn("grad_output", normalize_quant_config(config))
        with self.assertRaisesRegex(ValueError, "group_size=16"):
            normalize_quant_config({**config, "grad_output": {**NVFP4, "group_size": 8}})
        with self.assertRaisesRegex(ValueError, "quant.weight, quant.act and quant.grad_output"):
            normalize_quant_config({"weight": NVFP4, "act": None, "grad_output": NVFP4})

    def test_mxfp4_formats(self):
        from fake_quant_ops.quant.ops.mxfp import _quantize_mx

        torch.manual_seed(42)
        probe = torch.randn(4, 16) * 0.5
        probe[:, 0] = 3.8

        def quantize(value, minus_exp):
            return _quantize_mx(value.float(), scale_bits=8, elem_format="fp4_e2m1",
                                axes=-1, block_size=16, minus_exp=minus_exp)

        self.assertFalse(torch.equal(quantize(probe, None), quantize(probe, "auto")))
        for bit, minus_exp in (("mxfp4", None), ("mxfp4_auto", "auto")):
            with self.subTest(bit=bit):
                spec = {**NVFP4, "bit": bit}
                config = normalize_quant_config({"weight": spec, "act": spec, "grad_output": spec})
                linear = nn.Linear(16, 32)
                qat = QATLinear(linear, config)
                x = torch.randn(2, 3, 16, requires_grad=True)
                xq = quantize(x.detach(), minus_exp)
                wq = quantize(linear.weight.detach(), minus_exp)
                output = qat(x)
                torch.testing.assert_close(output, F.linear(xq, wq, linear.bias))

                dy = torch.randn_like(output)
                output.backward(dy)
                grad = dy.reshape(-1, 32)
                expected_dx = (quantize(grad, minus_exp) @ quantize(wq.T, minus_exp).T).reshape_as(x)
                expected_dw = quantize(F.pad(grad.T, (0, 10)), minus_exp) @ quantize(F.pad(xq.reshape(-1, 16).T, (0, 10)), minus_exp).T
                torch.testing.assert_close(x.grad, expected_dx)
                torch.testing.assert_close(linear.weight.grad, expected_dw)
                torch.testing.assert_close(linear.bias.grad, grad.sum(dim=0))

        with self.assertRaisesRegex(ValueError, "group_size=16"):
            normalize_quant_config({"weight": {**NVFP4, "bit": "mxfp4", "group_size": 32}})
        with self.assertRaisesRegex(ValueError, "same format"):
            normalize_quant_config({"weight": {**NVFP4, "bit": "mxfp4"}, "act": {**NVFP4, "bit": "mxfp4"}, "grad_output": {**NVFP4, "bit": "mxfp4_auto"}})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for autocast")
    def test_cuda_autocast_backward(self):
        torch.manual_seed(7)
        quantizer = NVFP4FakeQuantizer(NVFP4)
        linear = nn.Linear(16, 32, device="cuda")
        qat = QATLinear(linear, normalize_quant_config({"weight": NVFP4, "act": NVFP4, "grad_output": NVFP4}))
        x = torch.randn(2, 16, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = qat(x)
        dy = torch.randn_like(output)
        (output * dy).sum().backward()

        xq = quantizer.quantize(x.detach()).to(torch.bfloat16)
        wq = quantizer.quantize(linear.weight.detach()).to(torch.bfloat16)
        dy_dx = quantizer.quantize(dy.float()).to(torch.bfloat16)
        w_dx = quantizer.quantize(wq.T.float()).to(torch.bfloat16)
        dy_dw = quantizer.quantize(F.pad(dy.T.float(), (0, 14))).to(torch.bfloat16)
        x_dw = quantizer.quantize(F.pad(xq.T.float(), (0, 14))).to(torch.bfloat16)
        torch.testing.assert_close(x.grad, (dy_dx @ w_dx.T).float(), rtol=0.05, atol=0.05)
        torch.testing.assert_close(linear.weight.grad, (dy_dw @ x_dw.T).float(), rtol=0.05, atol=0.05)
        torch.testing.assert_close(linear.bias.grad, dy.sum(dim=0).float(), rtol=0.05, atol=0.05)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA devices are required for FSDP2")
    def test_fsdp2_backward(self):
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(_fsdp_worker, args=(port,), nprocs=2)


if __name__ == "__main__":
    unittest.main()


"""PYTHONPATH=/mnt/lm_data_afs/charles/codes/LightX2V:/mnt/lm_data_afs/charles/codes/LightX2V/lightx2v_train /mnt/lm_data_afs/charles/codes/LightX2V/.venv/bin/python /mnt/lm_data_afs/charles/codes/LightX2V/tests/test_train_quantization.py"""
