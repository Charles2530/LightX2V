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

```bash
# 在基线上叠加固定四步的 token 饱和（与渐进式分辨率互斥）
TOKEN_SATURATION=1 bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh --warmup
```

结束时打印一行 JSON：`request_s`（不含加载、编译和预热）、`peak_allocated_gib`、`peak_reserved_gib`。不加 `--warmup` 时，`request_s` 包含该尺寸首次编译的开销。内置 `warmup` 只覆盖 480×480 和 544×960，所以配置里保持 `warmup=false`，由 `--warmup` 按真实尺寸预热。

AdaLN cache 不存在时，入口会先用同一配置在进程内构建，已有 cache 不会被覆盖。也可以用 `tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py --config_json <基线配置>` 单独构建。

同时启动多个单卡实例时，为每个实例设置不同的 `MASTER_PORT`。

## 渐进式分辨率接口

基线配置已包含渐进式分辨率的全部键，`changing_resolution=false` 时不生效。打开后前 `changing_resolution_steps` 步在 `resolution_rate` 缩放后的 latent 网格上运行，再提升到完整分辨率。各键的含义、默认值和分阶段消融选项见 `configs/minimax_h3/changing_resolution/README.md`。

## Token 饱和（固定四步）

实现在 `lightx2v/models/schedulers/minimax_h3/token_saturation.py`，参考 Speculative Forcing（arXiv 2610.22347），按 H3 的双向单次打包序列做了改写。四步 sigma 调度不变，变化的是每一步参与前向的目标视频 token。

每次前向后，对每个目标视频 token 取干净草稿 `x0_hat = x + sigma * v`（H3 的速度指向数据端）。验证器无需训练：

- 稳定性：在 `token_saturation_verify_steps` 列出的前向之后，计算 `||x0_hat^(k) - x0_hat^(k-1)||`。
- 冗余（可选）：仅在第 0 次前向之后、且请求带关键帧时，计算草稿到最近关键帧 token 的距离。

分数除以全部目标 token 草稿范数的中位数（已冻结 token 用其提交值），在（latent 帧、token 行、token 列）上做可分离高斯平滑（反射填充，sigma=1 时 9 抽头），再与阈值比较。平滑只在活跃 token 上做归一化卷积 `smooth(score·mask) / smooth(mask)`，已冻结 token 和另一种分数都不参与。低于阈值的 token 被冻结为当前草稿，后续前向不再去噪它。

论文用 KV cache 保存已接受 token，在 768×1344×243 下约需 75 GB，这里改为把冻结 token 移出打包序列。落在 `token_saturation_context_stride` 空间格点上的冻结 token 保留为上下文行，每次前向重新计算，输出丢弃。上下文行的噪声水平由 `token_saturation_context_timestep` 决定：

- `current`（默认）：用冻结草稿和该 token 的初始噪声重新加噪到当前步的 sigma，放在目标段，使用目标时间步。上下文行与活跃行处在同一噪声水平，和稠密采样时模型看到的输入分布一致，AdaLN cache 的键也不变。
- `keyframe`：按关键帧的噪声增强加噪（sigma = 1 - 0.999），放在条件段，使用关键帧的条件时间步（0.999），复用 conditioned 请求的 AdaLN cache 条目。模型把它们当作干净条件，但训练时没见过这种位置分布，两种模式哪个更好需要实测。

最后一步结束后恢复完整布局，VAE 解码看到的仍是全部目标行。日志会打印每步的分数分位数、新冻结数、冻结比例，以及每个 token 的平均前向次数。

| 配置键 | 默认 | 含义 |
|---|---|---|
| `token_saturation` | `false` | 总开关，也可用 `--token_saturation` / `TOKEN_SATURATION=1` 打开 |
| `token_saturation_verify_steps` | `[1, 2]` | 做稳定性验证的前向序号，取值 1 到 `infer_steps - 2`；最后一次前向对所有 token 生效 |
| `token_saturation_tau_stab` | `0.2` | 稳定性阈值（归一化后） |
| `token_saturation_tau_red` | `null` | 关键帧冗余阈值，`null` 关闭 |
| `token_saturation_max_ratio` | `0.6` | 累计冻结比例上限，须小于 1 |
| `token_saturation_context_stride` | `2` | 冻结 token 保留为上下文的空间步长；0 全部移除，1 全部保留（之后仍按 `bucket` 取整） |
| `token_saturation_bucket` | `1024` | 活跃行数向上取整到该粒度；上下文行数取最近的倍数（等间隔去掉格点行，或补入格点外的冻结行），使目标视频行总数对齐。文本和音频行不在内，整条序列不一定对齐 |
| `token_saturation_smooth_sigma` | `[1.0, 1.0]` | 平滑的（时间, 空间）sigma |
| `token_saturation_context_timestep` | `"current"` | 上下文行的噪声水平：`current` 或 `keyframe`，见上文 |
| `token_saturation_oracle_dir` | `null` | 非空时进入 oracle 标定模式，结果写到该目录，也可用 `--token_saturation_oracle_dir` / `TOKEN_SATURATION_ORACLE_DIR` 设置 |
| `token_saturation_oracle_eps` | `[0.02, 0.05, 0.1]` | oracle 判定“已收敛”的相对误差阈值，可列多个 |
| `token_saturation_oracle_precision` | `0.95` | 标定阈值时要求的精度，取值 (0, 1] |

限制：只支持 Euler 四步、`feature_caching=NoCaching`、非 `veda_attn`、单卡（无 seq_parallel），不支持 Ref2AV、后缀重生成和 draft 前缀。打开后预热会跑满四步。

### Oracle 标定

```bash
# 跑稠密四步，记录每个 token 的草稿轨迹和验证分数（不冻结任何 token，输出视频与基线相同）
TOKEN_SATURATION_ORACLE_DIR=/path/to/oracle bash scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh
```

oracle 模式下不冻结任何 token，每步只记录草稿和分数。最后一步后，对每个 token 计算各步草稿到最终输出的相对误差 `rel_k = ||x0_hat^(k) - x0_final|| / median||x0_final||`，并写出两个文件，文件名为 `token_saturation_oracle_seed{seed}_{H}x{W}x{F}`：

- `.json`：
  - `rel_to_final_q10_q50_q90_q99`：每步 `rel_k` 的分位数。
  - `oracle`：每个 `eps` 下的 oracle 平均前向次数（`rel_k <= eps` 的最早一步加 1，否则为 4）和前向次数直方图，是该 `eps` 下的上限参考。
  - `calibration`：逐步独立（边际）标定。对每个稳定性步 1 到 `infer_steps - 2`（有关键帧时再加第 0 步的冗余分数），找出最大的 `tau`，使 `score < tau` 的 token 中 `rel_k <= eps` 的比例不低于 `oracle_precision`。只在同分组的边界处截断，接受比例、精度和召回率都按 `score < tau` 的实际集合重算。
  - `replay`：在稠密轨迹上按在线规则依次重放冻结，使用相同的尺度、带 mask 的平滑、`max_ratio` 和 `bucket`。`config` 用配置里的阈值；`calibrated` 对每个 `eps` 在仍活跃的 token 上逐步标定 `tau`。每项给出各步的 `tau` 和新冻结数、冻结比例、平均前向次数，以及冻结 token 相对最终输出的误差分位数和 `rel <= eps` 的比例。
- `.pt`：`rel_to_final`（步数 × token）以及每步原始和平滑后的分数，用于离线分析。

把 `replay.calibrated` 中选定 `eps` 各步的 `tau` 填入 `token_saturation_tau_stab` / `token_saturation_tau_red`（每类分数只有一个阈值，多个稳定性步时取较小的），再关闭 oracle 运行。重放假设冻结不改变其余 token 的轨迹，实际冻结后活跃 token 看到的上下文与稠密时不同，这部分偏差只能在 GPU 上对比 token 饱和输出和稠密输出来测。应在多个 prompt、seed 和任务（T2AV / I2AV / FL2AV）上标定后取保守值。加 `--warmup` 时预热请求同名，结果会被计时请求覆盖。

阈值尚未标定。以上默认值只是起点，需要在 GPU 上用 oracle 标定并对比基线的画质和耗时后调整；该路径还没有在 GPU 上运行过。

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

I2AV 实测发现 SwiGLU 融合 kernel 在 75,419 行、28,672 stride 下 int32 指针偏移溢出，已改为 int64 行偏移。另兼容了 Transformers 4.57.x/5.x 图像 token type API 的差异。
