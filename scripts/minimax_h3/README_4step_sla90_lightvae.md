# MiniMax-H3 四步 SLA90 + LightVAE 加速基线

单卡全分辨率基线，后续加速实验都以此为对照：

- 四步蒸馏：Euler，video/audio shift 6/3，无 CFG。DiT 使用已合并 Turbo-SLA 的 FP8 F16-accum 权重，不再加载 LoRA。
- 注意力：`dynamic_sparse_attn`，稀疏率 0.9，算子 Sage2。
- 视觉 patch Linear：`vision_patch_embed_linear=true`。Qwen3-VL patch Conv3d 的 stride 等于 kernel，因此等价于对展平 patch 做 Linear，权重不变。
- LightVAE：`video_vae_decoder_ckpt` 指向 LightVAE decoder 权重。原 VAE（含 FP8 encoder）先按原流程严格加载，再替换 `decoder` 和 `post_quant_conv`，encoder、latent 归一化、tiling 和并行 decode 保持不变。LightVAE decoder 用 FP16 Linear，敏感层用 FP32。该选项不支持 shared/block VAE offload。
- 其余：FP8-SGL 文本编码器，FP8 视频 VAE encoder（`cutlass_fp8_f16_accum`），AdaLN cache，DiT block offload + `dit_prepost_resident`，H.264 ultrafast/CRF 23。

| 文件 | 作用 |
|---|---|
| `configs/minimax_h3/distill/4steps/minimax_h3_fp8_4step_5090_sla90_lightvae.json` | 基线配置 |
| `scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh` | 启动脚本，设置路径和环境变量 |
| `scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.py` | Python 入口：缺失时构建 AdaLN cache，可选同尺寸预热，打印请求耗时和峰值显存 |

## 运行

先修改脚本顶部的 `lightx2v_path`、`model_path`、`quant_dir`、`lightvae_ckpt`，或用同名环境变量覆盖。默认 1344×768、243 帧、seed 0。

```bash
# 文生视频，预热一次后计时一次
bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh --warmup
```

```bash
# 图生视频（会经过视觉 patch Linear）；首尾帧再加 --last_frame_path
bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh --warmup --image_path /path/to/first_frame.png --prompt "A person smiles and waves at the camera."
```

```bash
# 在基线上叠加渐进式分辨率
CHANGING_RESOLUTION=1 bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh --warmup
```

结束时打印一行 JSON：`request_s`（不含加载、编译和预热）、`peak_allocated_gib`、`peak_reserved_gib`。不加 `--warmup` 时，`request_s` 包含该尺寸首次编译的开销。内置 `warmup` 只覆盖 480×480 和 544×960，所以配置里保持 `warmup=false`，由 `--warmup` 按真实尺寸预热。

AdaLN cache 不存在时，入口会先用同一配置在进程内构建，已有 cache 不会被覆盖。也可以用 `tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py --config_json <基线配置>` 单独构建。

同时启动多个单卡实例时，为每个实例设置不同的 `MASTER_PORT`。

## 渐进式分辨率接口

基线配置已包含渐进式分辨率的全部键，`changing_resolution=false` 时不生效。打开后前 `changing_resolution_steps` 步在 `resolution_rate` 缩放后的 latent 网格上运行，再提升到完整分辨率。各键的含义、默认值和分阶段消融选项见 `configs/minimax_h3/changing_resolution/README.md`。

## 权重与环境

| 内容 | 配置键 | 文件名 |
|---|---|---|
| 原始模型、tokenizer、audio VAE | `model_path` | MiniMax-H3 根目录 |
| 已合并四步 Turbo-SLA 的 FP8 DiT | `dit_quantized_ckpt` | `minimax_h3_dit_sla_merged_fp8_f16_accum.safetensors` |
| FP8 文本编码器 | `text_encoder_quantized_ckpt` | `minimax_h3_text_encoder_fp8_sgl.safetensors` |
| FP8 视频 VAE（encoder Conv3d FP8） | `video_vae_quantized_ckpt` | `minimax_h3_video_vae_fp8_f16_accum_encoder_conv_q21.safetensors` |
| LightVAE decoder | `video_vae_decoder_ckpt` | `lightvae-lite-h3-decoder.safetensors` |

依赖 SGL kernel、FlashInfer、SageAttention2、SpargeAttn（`spas_sage_attn`），以及带 SM120 FP8/F16-accum GEMM/Conv3d 的 LightX2V kernel。源码编译的扩展可通过 `EXTRA_PYTHONPATH` 加入 `PYTHONPATH`。

## 验证记录（2026-11-26，补丁版入口）

以下数据来自本次规范化之前的入口（LightVAE 和 patch Linear 以运行时补丁接入，加载的权重和计算方式与现在相同）。规范化后尚未在 GPU 上重测。

单张 RTX 5090，PyTorch 2.9.1+cu130、Transformers 4.57.6、Triton 3.5.1。1344×768、243 帧、seed 0，各预热一次后测一次请求：T2AV 36.84 s，I2AV 38.62 s。这是可运行性检查，不是三次中位数性能评测。两个视频都经 ffmpeg 完整解码。I2AV 中视觉 patch Linear 相对 Conv3d 的 L2 误差为 0.004957。

I2AV 实测发现 SwiGLU 融合 kernel 在 75,419 行、28,672 stride 下 int32 指针偏移溢出，已改为 int64 行偏移，回归测试为 `tests/test_h3_swiglu_large_stride.py`。另兼容了 Transformers 4.57.x/5.x 图像 token type API 的差异。
