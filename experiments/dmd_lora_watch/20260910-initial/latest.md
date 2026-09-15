# DMD action/video LoRA 训练观察

更新时间：2026-09-10T13:30:01+00:00；训练日志 step **1280**；LoRA 最新完整记录 step **1250**。

数据源：`lightx2v_train/runs/fastwam_robotwin_action_1step_dmd_lora_only_unfreeze_video/lora_monitor/from-000000000-34c2ae4f`；每 50 个 student 更新步记录；各分支 300 个 adapter layer。

## 当前需要关注

- WandB 复用旧 run，已确认丢弃指标：最新警告 step 1280 < 30001。本报告使用本地 CSV/log。
- 输出目录仍混有高步数旧 checkpoint：checkpoint-000026000, checkpoint-000028000, checkpoint-000030000。现有 save_total_limit=3 在保存前按步数裁剪，会混合处理两轮 checkpoint；auto-resume 也可能选中旧轮次。
- action 最近 5 个记录点中至少 3 次明显反转比例 ≥10%，需要结合损失/评测持续观察；这是筛查阈值，不是 loss 冲突证明。

## 最新分支比较

| 指标（逐层统计） | action | video |
| --- | ---: | ---: |
| 累计 ΔW 范数均值 | 0.684657 | 0.426649 |
| 累计 ΔW / W 均值 | 1.286% | 0.475% |
| 累计 ΔW / W 最大值 | 8.199% | 0.817% |
| 最近 50 步变化 / W 均值 | 0.336% | 0.085% |
| 最近 50 步变化 / W 最大值 | 1.491% | 0.134% |
| 相邻区间更新 cosine 均值 | -0.0995768 | 0.087038 |
| 相邻区间更新 cosine 中位数 | -0.0883011 | 0.0948131 |
| cosine < 0 比例 | 75.667% | 8.219% |
| cosine < −0.2 比例 | 25.667% | 0.000% |
| 未变化比例 | 0.000% | 2.667% |
| 裁剪前 A/B 梯度范数均值 | 0.000186612 | 0.000674115 |
| 非有限权重统计层数 | 0 | 0 |
| 非有限梯度层数 | 0 | 0 |

范数为 Frobenius；方向比例仅以非零且有限的区间更新为分母。两分支 rank/LR 不同：action rank128、LR 1e-4；video rank8、LR 1e-5，不能直接用 A/B 梯度大小比较其作用强弱。

## 最近记录

| step | action ΔW/W | video ΔW/W | action cosine | video cosine | action 反转 | video 反转 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 800 | 0.844% | 0.321% | -0.0785882 | 0.0706133 | 16.000% | 0.342% |
| 850 | 0.883% | 0.340% | 0.000337233 | 0.156371 | 11.000% | 0.000% |
| 900 | 0.965% | 0.358% | -0.0330968 | 0.09086 | 13.667% | 0.000% |
| 950 | 1.003% | 0.383% | -0.080379 | 0.120122 | 18.000% | 0.000% |
| 1000 | 1.029% | 0.398% | -0.0423963 | 0.103757 | 12.333% | 0.000% |
| 1050 | 1.096% | 0.412% | -0.0940853 | 0.0745122 | 16.333% | 0.000% |
| 1100 | 1.146% | 0.422% | -0.0222236 | 0.0711344 | 11.000% | 0.685% |
| 1150 | 1.166% | 0.440% | -0.0651534 | 0.082657 | 14.667% | 0.342% |
| 1200 | 1.251% | 0.458% | -0.0952052 | 0.1198 | 18.667% | 0.000% |
| 1250 | 1.286% | 0.475% | -0.0995768 | 0.087038 | 25.667% | 0.000% |

## Loss 与诊断评测

| step 区间 | 已记录点数 | DMD 均值 | fake 均值 | student grad 均值 | video grad 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–250 | 26 | 0.160933 | 0.00579946 | 0.0114654 | 0.0134846 |
| 251–500 | 25 | 0.228817 | 0.0025512 | 0.007312 | 0.022768 |
| 501–750 | 25 | 0.214685 | 0.00331648 | 0.006684 | 0.01706 |
| 751–1000 | 25 | 0.187028 | 0.0033082 | 0.005868 | 0.01586 |
| 1001–1250 | 25 | 0.189351 | 0.00372708 | 0.0058 | 0.017128 |
| 1251–1280 | 3 | 0.173822 | 0.00360833 | 0.00533333 | 0.0245333 |

| eval step | student–teacher L1 | student–GT L1 |
| ---: | ---: | ---: |
| 500 | 0.010371 | 0.014475 |
| 1000 | 0.010956 | 0.015015 |

DMD 数值是随 fake/teacher 信号变化的训练代理目标，不等同于固定验证误差。此配置每 rank 仅 4 个诊断样本；L1 不能替代 RoboTwin 闭环成功率。teacher 也使用当前共享 video condition，因此 student–teacher L1 不是相对原始完整冻结 FastWAM 的固定参照。

## 层级证据

**action：** 最近区间未变化 0/300 层。


最负的 5 个 cosine：

- `base_model.model.blocks.29.cross_attn.o`：cos=-0.594925；区间变化/W=0.382%。
- `base_model.model.blocks.28.ffn.2`：cos=-0.556338；区间变化/W=0.311%。
- `base_model.model.blocks.29.ffn.2`：cos=-0.52923；区间变化/W=0.273%。
- `base_model.model.blocks.29.self_attn.o`：cos=-0.50337；区间变化/W=0.325%。
- `base_model.model.blocks.29.ffn.0`：cos=-0.501483；区间变化/W=0.287%。

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

- `base_model.model.blocks.28.self_attn.v`：cos=-0.127697；区间变化/W=0.064%。
- `base_model.model.blocks.22.cross_attn.o`：cos=-0.104422；区间变化/W=0.123%。
- `base_model.model.blocks.3.cross_attn.q`：cos=-0.0901489；区间变化/W=0.079%。
- `base_model.model.blocks.29.self_attn.v`：cos=-0.0874107；区间变化/W=0.030%。
- `base_model.model.blocks.27.self_attn.o`：cos=-0.0792139；区间变化/W=0.074%。

video 第 29 block（最后一层）只有 self-attn K/V 向 action 提供缓存；该层 Q、O、cross-attn、FFN 的输出没有后续 video layer 消费。因而目前其余 8 个 adapter 的零更新与计算图结构一致，不表示 video 分支整体未学习。证据：`mot.py:272–313` 的逐层 K/V 缓存与末层输出用途；当前 anchor 在初始化处梯度为零，video weight_decay=0。

## 解读边界与后续

video 有效权重非零且持续变化可以证明参与参数学习；持续低反转支持在 50 步时间尺度上相对稳定，不能证明提高任务成功率。action 的负 cosine 提示更新在反复修正，不能据此归因为 action/video 目标冲突。

先关注 step 1500/2000 诊断是否持续恶化、action 反转是否继续增多、video 是否出现更新尖峰或 NaN。比较任务效果需要本轮 checkpoint 的闭环评测和匹配训练预算的 action-only 对照。

本报告由独立只读观察进程更新，不修改训练配置、权重或 optimizer，不发送外部消息。观察进程停止或当前执行环境消失后不会继续更新；它不会自动向聊天推送通知。
