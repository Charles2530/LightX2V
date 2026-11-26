# H3-20 baseline

入口是仓库根目录的 `baseline_h3.sh`。四步蒸馏、Euler、video/audio shift 6/3、90% 稀疏 Sage2 SLA、视觉 patch Conv3d 等价替换 Linear、官方 LightVAE 12 层 decoder；单卡全分辨率。

DiT 使用已合并 Turbo-SLA 的 FP8/F16-accum 权重，不要再次加载同一 LoRA。文本编码器为 FP8-SGL，视频 VAE encoder 为 FP8；LightVAE decoder 使用 FP16 Linear 和 FP32 敏感层。后续实验的 VAE 核心 FP8、渐进分辨率、Ulysses 窗口调优均未启用。编码为 H.264 ultrafast/CRF 23。

## 本机复现

在仓库根目录执行，先用 `nvidia-smi` 确认所选 GPU 空闲。默认 1344×768、243 帧、seed 0。

```bash
# 检查路径、依赖导入和配置，不加载模型
CUDA_VISIBLE_DEVICES=0 ./baseline_h3.sh --check

# 文生视频，预热一次后记录一次请求
CUDA_VISIBLE_DEVICES=0 ./baseline_h3.sh --warmup \
  --prompt 'A cinematic starship flies through a colorful nebula, engines humming.' \
  --output save_results/h3_20/t2av.mp4

# 图生视频，实际执行并验证 patch Linear
CUDA_VISIBLE_DEVICES=0 ./baseline_h3.sh --warmup \
  --image /absolute/path/first_frame.png \
  --prompt 'A person smiles and waves at the camera.' \
  --output save_results/h3_20/i2av.mp4

# 首尾帧：图生视频命令增加 --last-frame /absolute/path/last_frame.png
ffmpeg -v error -xerror -i save_results/h3_20/t2av.mp4 -f null -
```

默认解释器是 `.venv_h3_runtime/bin/python`。脚本采用当前 `LightX2VPipeline(config)` API，自动生成缺失的 AdaLN cache，不覆盖已有 cache。首次缓存生成、权重加载和编译可能较久；`--warmup` 排除首轮请求的编译开销。已有兼容缓存可通过 `H3_ADALN_CACHE` 指定，例如本机 `experiments/h3_progressive/adaln_cache`。

产物包括 MP4、同名 `.config.json` 和 `.metrics.json`，同目录另有 `lightvae_integration.json`。图片任务首次调用视觉 patch 时写入 `vision_patch_validation.json`，检查实际输入相对 Conv3d 的 L2 误差小于 0.02。T2AV 不使用视觉 tower，因此不会生成该文件。不同实验应使用独立目录，避免公共诊断文件覆盖。

并发启动独立单卡实例时，分别设置不同的 `MASTER_PORT`，例如 29680 和 29681，避免分布式初始化端口冲突。

## 权重与环境

仓库不包含权重、虚拟环境和编译后的 CUDA 二进制。默认模型根目录是 `/mnt/afs_1/charles/models`。

| 内容 | 默认相对路径 |
|---|---|
| 原始模型、tokenizer、audio VAE | MiniMax-H3-root/ |
| 已合并四步 Turbo-SLA 的 FP8 DiT | MiniMax-H3-quant/minimax_h3_dit_sla_merged_fp8_f16_accum.safetensors |
| FP8 文本编码器 | MiniMax-H3-quant/minimax_h3_text_encoder_fp8_sgl.safetensors |
| FP8 原版 VAE encoder | MiniMax-H3-quant/minimax_h3_video_vae_fp8_f16_accum_encoder_conv_q21.safetensors |
| LightVAE decoder | LightVAE/lightvae-lite-h3-decoder.safetensors |

迁移时可设置 `H3_MODELS_DIR`、`H3_MODEL_ROOT`、`H3_QUANT_DIR`；单独覆盖用 `H3_DIT_CKPT`、`H3_TEXT_CKPT`、`H3_VIDEO_VAE_CKPT`、`H3_LIGHTVAE_CKPT`。包含 Transformers 4.57.x 图像输入的 token type 兼容修复。切换解释器用 `H3_PYTHON`。

本机环境为 RTX 5090、PyTorch 2.9.1+cu130、Transformers 4.57.6、Triton 3.5.1、safetensors 0.8.0、PyAV 18.1.0。另外需要 SGL kernel、FlashInfer、SageAttention2、SpargeAttn (`spas_sage_attn`)，以及含 SM120 FP8/F16-accum GEMM/Conv3d 的 LightX2V kernel。

本机源码编译扩展从 `vendor/lxk_clean`、`vendor/sparge_attn_cu130_current_stream`、`/mnt/afs_1/charles/experiments/minimax_h3/report_20261021/runtime` 和 `research_20261021/flashinfer031_runtime` 导入。这些机器环境不随 baseline 提交。新机器需先安装对应 CUDA 扩展，并运行 `--check`；可用 `H3_RUNTIME_ROOT` 改变 runtime 根目录，或 `H3_EXTRA_PYTHONPATH` 指定其他已构建扩展位置。仅复制脚本不足以在缺少权重或扩展的机器上运行。

## 验证记录（2026-11-26）

单张 RTX 5090、1344×768、243 帧、seed 0，各预热一次后测一次请求。T2AV 使用远端 HEAD 生产模块隔离本地实验修改，36.84 秒；I2AV 在下面的 int64 修复后为 38.62 秒。均为入口可运行性检查，非三次中位数性能评测。两者正式视频及预热视频均经 ffmpeg 全段音视频解码。I2AV 的视觉 patch 相对 L2 误差为 0.004957，小于 0.02 阈值。

I2AV 实测发现已有 SwiGLU 融合 kernel 在 75,419 行、28,672 stride 下发生 int32 指针偏移溢出。提交将行偏移计算改为 int64；`tests/test_h3_swiglu_large_stride.py` 对旧版失败、修复版通过。此修复不改变激活和 FP8 量化计算。Transformers 4.57.x/5.x 的图像 token type API 差异也在提交中兼容。
