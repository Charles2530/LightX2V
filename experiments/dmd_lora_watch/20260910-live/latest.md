# DMD action/video LoRA 训练观察

更新时间：2026-09-10T15:25:52+00:00；训练日志 step **2000**；LoRA 最新完整记录 step **2000**。

数据源：`/mnt/miaohua/charles/codes/LightX2V_fastwam/lightx2v_train/runs/fastwam_robotwin_action_1step_dmd_lora_only_unfreeze_video/lora_monitor/from-000000000-34c2ae4f`；每 50 个 student 更新步记录；各分支 300 个 adapter layer。

## 当前需要关注

- WandB 复用旧 run，已确认丢弃指标：最新警告 step 2000 < 30001。本报告使用本地 CSV/log。
- 输出目录仍混有高步数旧 checkpoint：checkpoint-000028000, checkpoint-000030000。现有 save_total_limit=3 在保存前按步数裁剪，会混合处理两轮 checkpoint；auto-resume 也可能选中旧轮次。
- action 最近 5 个记录点中至少 3 次明显反转比例 ≥10%，需要结合损失/评测持续观察；这是筛查阈值，不是 loss 冲突证明。

## 最新分支比较

| 指标（逐层统计） | action | video |
| --- | ---: | ---: |
| 累计 ΔW 范数均值 | 0.999528 | 0.579583 |
| 累计 ΔW / W 均值 | 1.880% | 0.647% |
| 累计 ΔW / W 最大值 | 10.930% | 1.151% |
| 最近 50 步变化 / W 均值 | 0.380% | 0.087% |
| 最近 50 步变化 / W 最大值 | 1.748% | 0.133% |
| 相邻区间更新 cosine 均值 | -0.011944 | 0.0699555 |
| 相邻区间更新 cosine 中位数 | -0.000794642 | 0.0815788 |
| cosine < 0 比例 | 50.000% | 13.014% |
| cosine < −0.2 比例 | 8.667% | 0.342% |
| 未变化比例 | 0.000% | 2.667% |
| 裁剪前 A/B 梯度范数均值 | 0.000289184 | 0.0019544 |
| 非有限权重统计层数 | 0 | 0 |
| 非有限梯度层数 | 0 | 0 |

范数为 Frobenius；方向比例仅以非零且有限的区间更新为分母。两分支 rank/LR 不同：action rank128、LR 1e-4；video rank8、LR 1e-5，不能直接用 A/B 梯度大小比较其作用强弱。

## 最近记录

| step | action ΔW/W | video ΔW/W | action cosine | video cosine | action 反转 | video 反转 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1550 | 1.533% | 0.551% | -0.0529238 | 0.0959754 | 13.667% | 0.000% |
| 1600 | 1.554% | 0.564% | -0.022727 | 0.0468992 | 10.333% | 0.000% |
| 1650 | 1.603% | 0.573% | -0.070715 | 0.0892943 | 15.000% | 0.342% |
| 1700 | 1.637% | 0.586% | -0.076609 | -0.00360258 | 17.333% | 1.370% |
| 1750 | 1.670% | 0.599% | -0.0309924 | 0.126172 | 10.000% | 0.000% |
| 1800 | 1.717% | 0.606% | -0.0417983 | 0.034349 | 13.667% | 0.685% |
| 1850 | 1.756% | 0.618% | -0.0588636 | 0.10517 | 12.000% | 0.000% |
| 1900 | 1.785% | 0.628% | -0.0998845 | 0.0925559 | 19.333% | 0.000% |
| 1950 | 1.826% | 0.636% | -0.0585142 | 0.0992287 | 14.000% | 0.342% |
| 2000 | 1.880% | 0.647% | -0.011944 | 0.0699555 | 8.667% | 0.342% |

## Loss 与诊断评测

| step 区间 | 已记录点数 | DMD 均值 | fake 均值 | student grad 均值 | video grad 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–250 | 26 | 0.160933 | 0.00579946 | 0.0114654 | 0.0134846 |
| 251–500 | 25 | 0.228817 | 0.0025512 | 0.007312 | 0.022768 |
| 501–750 | 25 | 0.214685 | 0.00331648 | 0.006684 | 0.01706 |
| 751–1000 | 25 | 0.187028 | 0.0033082 | 0.005868 | 0.01586 |
| 1001–1250 | 25 | 0.189351 | 0.00372708 | 0.0058 | 0.017128 |
| 1251–1500 | 25 | 0.1792 | 0.00388452 | 0.0053 | 0.018508 |
| 1501–1750 | 25 | 0.173237 | 0.00380248 | 0.005 | 0.014764 |
| 1751–2000 | 25 | 0.17944 | 0.00383572 | 0.005392 | 0.019452 |

| eval step | student–teacher L1 | student–GT L1 |
| ---: | ---: | ---: |
| 500 | 0.010371 | 0.014475 |
| 1000 | 0.010956 | 0.015015 |
| 1500 | 0.011566 | 0.014636 |
| 2000 | 0.012929 | 0.01655 |

DMD 数值是随 fake/teacher 信号变化的训练代理目标，不等同于固定验证误差。此配置每 rank 仅 4 个诊断样本；L1 不能替代 RoboTwin 闭环成功率。teacher 也使用当前共享 video condition，因此 student–teacher L1 不是相对原始完整冻结 FastWAM 的固定参照。

## 层级证据

**action：** 最近区间未变化 0/300 层。


最负的 5 个 cosine：

- `base_model.model.blocks.29.ffn.2`：cos=-0.621545；区间变化/W=0.287%。
- `base_model.model.blocks.29.cross_attn.o`：cos=-0.542795；区间变化/W=0.359%。
- `base_model.model.blocks.28.ffn.2`：cos=-0.508649；区间变化/W=0.293%。
- `base_model.model.blocks.29.ffn.0`：cos=-0.503146；区间变化/W=0.289%。
- `base_model.model.blocks.29.self_attn.o`：cos=-0.477849；区间变化/W=0.349%。

**video：** 最近区间未变化 8/300 层。

- `base_model.model.blocks.29.self_attn.q`
- `base_model.model.blocks.29.self_attn.o`
- `base_model.model.blocks.29.cross_attn.q`
- `base_model.model.blocks.29.cross_attn.k`
- `base_model.model.blocks.29.cross_attn.v`
- `base_model.model.blocks.29.cross_attn.o`
- `base_model.model.blocks.29.ffn.0`
- `base_model.model.blocks.29.ffn.2`

最负的 5 个 cosine：

- `base_model.model.blocks.28.cross_attn.v`：cos=-0.212208；区间变化/W=0.089%。
- `base_model.model.blocks.28.ffn.2`：cos=-0.173826；区间变化/W=0.041%。
- `base_model.model.blocks.22.cross_attn.o`：cos=-0.143172；区间变化/W=0.125%。
- `base_model.model.blocks.29.self_attn.v`：cos=-0.138244；区间变化/W=0.037%。
- `base_model.model.blocks.27.self_attn.v`：cos=-0.117439；区间变化/W=0.071%。

video 第 29 block（最后一层）只有 self-attn K/V 向 action 提供缓存；该层 Q、O、cross-attn、FFN 的输出没有后续 video layer 消费。因而目前其余 8 个 adapter 的零更新与计算图结构一致，不表示 video 分支整体未学习。证据：`mot.py:272–313` 的逐层 K/V 缓存与末层输出用途；当前 anchor 在初始化处梯度为零，video weight_decay=0。

## 解读边界与后续

video 有效权重非零且持续变化可以证明参与参数学习；持续低反转支持在 50 步时间尺度上相对稳定，不能证明提高任务成功率。action 的负 cosine 提示更新在反复修正，不能据此归因为 action/video 目标冲突。

先关注 step 1500/2000 诊断是否持续恶化、action 反转是否继续增多、video 是否出现更新尖峰或 NaN。比较任务效果需要本轮 checkpoint 的闭环评测和匹配训练预算的 action-only 对照。

本报告由独立只读观察进程更新，不修改训练配置、权重或 optimizer，不发送外部消息。观察进程停止或当前执行环境消失后不会继续更新；它不会自动向聊天推送通知。
