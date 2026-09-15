# FastWAM DMD 1-step LoRA 更新监控

监控分别统计 student 的 **action LoRA** 和共享视觉分支的 **video LoRA**。fake-score 模型的 LoRA 不计入这两组。监控仅读取参数和已有梯度，不增加 backward，不改变 loss、optimizer、裁剪、调度器或梯度同步逻辑。

## 启用方式

当前启动配置 `configs/train/fastwam_action_dmd/robotwin_action_1step_dmd_lora_only_unfreeze_video.yaml` 已启用以下设置，原有 torchrun 命令无需修改：

```yaml
logging:
  train_log_every_iters: 10
  lora_monitor:
    enabled: true
    every_n_steps: 50
    reversal_cos_threshold: -0.2
    zero_tolerance: 1.0e-12
    save_snapshots: true
    keep_last_snapshots: 2
```

`every_n_steps` 按 student/外层训练迭代计数，不按 fake 更新次数、microbatch 或 GPU 数量计数。需要每 10 步采集时改成 `10`。其他配置未添加此块时默认关闭；比较 action-only 时，将同一配置块加入该实验的 `logging` 下即可。

两节点训练仅 global rank 0 计算统计和写文件。两个节点都需要使用更新后的代码。已运行的进程不会自动加载这些修改，需要正常重启或从 checkpoint 恢复；此次修改不启动训练，也不自动改变 resume 设置。

## 统计定义

对一个 LoRA layer/adapter，记原始冻结权重为 W，第 k 次记录的有效增量为 D_k = s_k B_k A_k，区间更新为 U_k = D_k − D_(k−1)。这里 s 使用 PEFT layer 实际的 scaling，不假设固定为 alpha/r。

下表的范数均为将矩阵展平后的 L2，即 **Frobenius 范数**，不是矩阵最大奇异值定义的谱范数。

| CSV 字段 | 定义或含义 |
| --- | --- |
| `delta_w_norm` | ‖D_k‖_F，当前 LoRA 对原始权重的实际增量 |
| `relative_delta_w_norm` | ‖D_k‖_F / ‖W‖_F |
| `interval_update_norm` | ‖U_k‖_F，相比上次记录的变化 |
| `relative_interval_update_norm` | ‖U_k‖_F / ‖W‖_F |
| `update_cosine` | ⟨U_k,U_(k−1)⟩_F / (‖U_k‖_F ‖U_(k−1)‖_F) |
| `negative_direction` | 当前有效 cosine < 0 |
| `reversed_direction` | 当前有效 cosine < `reversal_cos_threshold`，默认 −0.2 |
| `reversal_fraction_so_far` | 本 layer 在本次进程中累计明显反转次数 / 有效方向比较次数 |
| `gradient_norm` | sqrt(‖grad(A)‖² + ‖grad(B)‖²)，不乘 scale |
| `gradient_tensors_present` | A/B 中有 `.grad` 的张量数量，区分缺失梯度与数值为零 |
| `active` / `trainable` | adapter 是否启用、A/B 是否有可训练参数 |

cosine 比较的是两段**实际参数变化**，不是 D_k 和 D_(k−1) 的夹角，也不是两个 loss 的梯度夹角。间隔 50 时，它比较累计 50 步的变化；期间可能发生而后抵消的逐步震荡无法从这个指标看出。需要更细的时间分辨率时减小间隔。

初始化时先记录一次 baseline；从 step 0 开始且间隔 50 时，step 50 获得第一段更新，step 100 才首次可能得到有效 cosine。任何一段更新范数不超过 `zero_tolerance` 时 cosine 留空，不当作 0，也不计入反转比例。原始 W 范数为 0 时相对量留空。非有限值留空并单独计数，不混入均值。

action 梯度在梯度累积及 DDP 同步后、原有裁剪和 optimizer.step 前采集；video 梯度在原有显式 all-reduce 后、裁剪和 optimizer.step 前采集。梯度范数是**记录步那一次更新**的梯度，不是整个记录区间的平均梯度。video 同步会将本地缺失梯度补零，因而这里的 missing 指标不能判断每个 rank 原先是否都参与了计算。

## 输出文件和 WandB

每次进程创建独立目录：

```text
<training.output_dir>/lora_monitor/from-<初始step>-<session>/
  metadata.json
  layers.csv
  summary.csv
  step-000000000.pt
  step-000000050.pt
  ...
```

- `layers.csv`：每个记录步、每个 layer/adapter 一行，包含上表全部指标。
- `summary.csv`：每个记录步的 action/video 各一行。范数、cosine、梯度范数和累计反转比例分别汇总 mean、median、max、有效 count；各 layer 等权，不按参数数量加权。
- `metadata.json`：本次配置、baseline step、layer 名称和统计约定。
- `.pt`：保存当时的 CPU FP32 A/B、scale、active 状态和方向计数，可重建 D_k；不是可直接恢复训练的完整 checkpoint。

默认 `keep_last_snapshots: 2` 只保留本次监控目录最近两份 `.pt`，CSV 保留全历史。设置 `0` 保留所有快照；`save_snapshots: false` 仅关闭快照写盘，仍计算全部统计并保留内存中的两次历史。监控内存也使用低秩因子，不保存完整大矩阵 BA。

恢复训练时从加载后的权重建立新 baseline，不沿用之前进程的方向历史；例如恢复 step 53、间隔 10 时首次区间为 53→60，下一次为 60→70。解释速度/更新大小时需注意首段长度不同，不能把缺失的恢复前历史当成零更新。

WandB 打开时，汇总记录自动进入现有 run；关闭时终端和 CSV 仍可用。建议绘制以下曲线，并将 `video` 替换为 `action` 同时比较：

```text
lora_monitor/video/delta_w_norm_mean
lora_monitor/video/relative_delta_w_norm_mean
lora_monitor/video/relative_interval_update_norm_mean
lora_monitor/video/relative_interval_update_norm_max
lora_monitor/video/gradient_norm_mean
lora_monitor/video/update_cosine_mean
lora_monitor/video/update_cosine_median
lora_monitor/video/valid_direction_count
lora_monitor/video/negative_direction_fraction
lora_monitor/video/reversal_fraction
lora_monitor/video/reversal_fraction_so_far_mean
lora_monitor/video/unchanged_fraction
lora_monitor/video/missing_gradient_fraction
lora_monitor/video/nonfinite_weight_layer_count
lora_monitor/video/nonfinite_gradient_layer_count
lora_monitor/seconds
```

`reversal_fraction` 是当前记录步中明显反转 layer 的比例，分母只包含有效 cosine 的 layer；`negative_direction_fraction` 使用较宽松的 < 0 阈值。`reversal_fraction_so_far_mean` 则平均各层的累计反转频率，两者不要混淆。

action-only 没有可训练 video 分支时，`video/has_lora=0`、`layer_count=0`，其均值指标不输出。这表示该组没有监控对象，不能解释成“video LoRA 学不动”。

## 如何判断三种情况

| 现象 | 优先查看的证据 | 可以得到的结论 |
| --- | --- | --- |
| video 基本不变 | 确认有 active/trainable layer；连续多次区间相对变化接近零，同时看梯度是否缺失、为零或很小 | 当前 video adapter 的有效权重几乎未更新；进一步区分梯度未到达、更新太小或已趋于收敛 |
| video 持续稳定改变 | 区间相对变化持续非零，梯度有限，cosine 多数不为负，反转比例较低，max 无突增 | video 有实际学习更新，观测时间尺度上的方向较稳定 |
| video 大幅反复变化 | 相对区间变化较大或出现尖峰，同时有效 cosine 经常为负、反转比例持续偏高 | 存在值得排查的优化震荡；结合学习率、梯度、DMD loss 和任务表现判断 |

不要只看累计 ‖D_k‖：它很大但 U_k≈0 可能表示已停止更新；A/B 分别变化但 BA 不变也不会算成有效更新。梯度范数依赖 A/B 的参数化，因此判断学习幅度以 ΔW 为主。

“接近零”“较大”没有跨模型通用阈值。应在相同训练步、数据和记录间隔下，与 action 分支及另一组实验的相对变化分布比较。稳定更新不能单独证明对任务有用；较大负 cosine 也不能证明 video 与 action/DMD 目标冲突，因为 batch 噪声、Adam 动量等同样可能导致反转。要检验 loss 冲突，需要另行计算同一输入上各 loss 对同一参数集合的梯度夹角。

## 成本和验证范围

统计通过恒等式 ⟨LR,UV⟩_F = sum((LᵀU) ⊙ (RVᵀ)) 计算低秩积的范数和夹角，使用 FP64 Gram 运算；区间差先在因子上做差，减少相近大范数相减引起的误差。保留两份 CPU 因子历史，额外内存随 LoRA 参数量增长。仍会增加 rank 0 的计算、CPU/GPU 传输和磁盘 I/O，其他 rank 可能在后续同步时等待。`lora_monitor/seconds` 记录权重统计及写盘时间，不含此前梯度采集和初始化扫描原始 W 的耗时；不能将它当作完整训练开销。

单元测试覆盖显式 BA 数值对照、方向反转、因子重参数化、微小更新、scale 改变、缺失梯度、非有限值、快照保留和恢复 baseline。小模型使用真实 trainer 训练循环和 PEFT LoRA，对比监控开关后的 action/fake/video 参数及 Torch RNG 逐位一致，并验证梯度在同步后、裁剪前采集。此验证不代替完整的两节点 16 GPU 训练；实际 video 学习状态要以新训练产生的日志为准。

2026-09-10 验证：新增监控测试及现有 `test_fastwam_action_dmd_video_unfreeze.py` 合计 **24 项通过**，修改的 Python 文件通过 Ruff 检查。额外在 H100 上对 3072×3072、rank 128 的 FP64 合成因子验证：低秩范数与显式 BA/区间差的相对误差小于 1e-9，cosine 绝对误差约 9.15e-16。预热后 20 次均值中，单层三次范数与一次内积计算约 0.27 ms；这不包含因子复制、差分构造、梯度采集、快照 I/O，也不能推算为完整模型训练的开销比例。
