"""Read existing DMD logs; write a local report without touching training state."""

import argparse
import csv
import io
import itertools
import json
import math
import os
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path


def utc():
    return datetime.now(UTC).isoformat(timespec="seconds")


def complete_text(path):
    text = path.read_text(errors="replace")
    return text[: text.rfind("\n") + 1]


def rows(path):
    return list(csv.DictReader(io.StringIO(complete_text(path))))


def number(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean(records, key):
    values = [number(r, key) for r in records]
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def fmt(value, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.3f}%" if percent else f"{value:.6g}"


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def inspect(run, session, output):
    summary = rows(session / "summary.csv")
    common_steps = set.intersection(*[{int(r["step"]) for r in summary if r["branch"] == b} for b in ("action", "video")])
    step = max(common_steps)
    summary = [r for r in summary if int(r["step"]) <= step]
    latest = {b: next(r for r in reversed(summary) if r["branch"] == b and int(r["step"]) == step) for b in ("action", "video")}
    # The layer history can contain hundreds of thousands of rows. Retain
    # only the already completed summary step rather than loading all history.
    with (session / "layers.csv").open(newline="") as handle:
        layer_rows = [r for r in csv.DictReader(handle) if int(r["step"]) == step]
    text = complete_text(run / "train.log")
    # train.log is append-only across restarts. Exclude previous experiments.
    start = text.rfind("[train] start")
    if start >= 0:
        text = text[start:]
    train = []
    evaluations = []
    for line in text.splitlines():
        match = re.search(r"iter=(\d+)/\d+.*dmd=([\d.e+-]+) fake=([\d.e+-]+) student_grad=([\d.e+-]+) fake_grad=([\d.e+-]+) video_grad=([\d.e+-]+)", line)
        if match:
            train.append(dict(zip(("step", "dmd", "fake", "student_grad", "fake_grad", "video_grad"), map(float, match.groups()))))
        match = re.search(r"\[eval\] iter=(\d+) student_teacher_l1=([\d.e+-]+) student_gt_l1=([\d.e+-]+)", line)
        if match:
            evaluations.append(dict(zip(("step", "student_teacher_l1", "student_gt_l1"), map(float, match.groups()))))
    training_step = int(train[-1]["step"]) if train else step
    age = max(0, time.time() - (run / "train.log").stat().st_mtime)
    alerts = []
    if len(evaluations) >= 4:
        recent_eval = evaluations[-4:]
        if all(right["student_gt_l1"] > left["student_gt_l1"] for left, right in itertools.pairwise(recent_eval)):
            alerts.append(
                f"GT L1 从 step {int(recent_eval[0]['step'])} 到 {int(recent_eval[-1]['step'])} 连续三次升高；需要扩大验证样本并核对闭环表现，不能由此单独归因于 video LoRA。"
            )
    if len(evaluations) >= 2:
        before, after = evaluations[-2:]
        keys = ("student_teacher_l1", "student_gt_l1")
        if all(before[k] > 0 and after[k] > 1.1 * before[k] for k in keys):
            changes = [after[k] / before[k] - 1 for k in keys]
            alerts.append(
                f"诊断 step {int(before['step'])}→{int(after['step'])}：teacher L1 上升 {changes[0]:.1%}，GT L1 上升 {changes[1]:.1%}。两项均超过 10% 的观察阈值；小样本诊断不能替代闭环评测。"
            )
    if re.search(r"Traceback \(most recent call last\)|\| ERROR \||(?:dmd|fake|student_grad|video_grad)=(?:nan|[+-]?inf)(?:\s|$)", text, re.IGNORECASE):
        alerts.append("本次训练日志发现 traceback、ERROR 或非有限 loss/gradient，请查看原始 train.log 定位。")
    wandb_runs = list((run / "wandb").glob("run-*"))
    if wandb_runs:
        wb = max(wandb_runs, key=lambda p: p.stat().st_mtime)
        wb_output = wb / "files/output.log"
        if wb_output.exists():
            warnings = re.findall(r"Tried to log to step (\d+) that is less than the current step (\d+)", complete_text(wb_output))
            if warnings:
                alerts.append(f"WandB 复用旧 run，已确认丢弃指标：最新警告 step {warnings[-1][0]} < {warnings[-1][1]}。本报告使用本地 CSV/log。")
    old_checkpoints = []
    for p in run.glob("checkpoint-*"):
        if (p / "training_state.pt").exists() and int(p.name.split("-")[-1]) > training_step:
            old_checkpoints.append(p.name)
    if old_checkpoints:
        alerts.append(
            "输出目录仍混有高步数旧 checkpoint：" + ", ".join(sorted(old_checkpoints)) + "。现有 save_total_limit=3 在保存前按步数裁剪，会混合处理两轮 checkpoint；auto-resume 也可能选中旧轮次。"
        )
    if age > 900:
        alerts.append(f"训练日志超过 15 分钟未更新（{age / 60:.1f} 分钟）；需检查训练节点状态，本进程仅观察共享文件。")
    if training_step - step > 100:
        alerts.append("LoRA CSV 比训练落后超过 100 步，需要检查监控输出。")
    for b, r in latest.items():
        if sum(number(r, k) or 0 for k in ("nonfinite_weight_layer_count", "nonfinite_gradient_layer_count")):
            alerts.append(f"{b} 出现非有限权重统计或梯度。")
        recent = [r for r in summary if r["branch"] == b and step - 200 <= int(r["step"]) <= step]
        if len(recent) >= 3 and sum((number(r, "reversal_fraction") or 0) >= 0.1 for r in recent) >= 3:
            alerts.append(f"{b} 最近 5 个记录点中至少 3 次明显反转比例 ≥10%，需要结合损失/评测持续观察；这是筛查阈值，不是 loss 冲突证明。")
    lines = [
        "# DMD action/video LoRA 训练观察",
        "",
        f"更新时间：{utc()}；训练日志 step **{training_step}**；LoRA 最新完整记录 step **{step}**。",
        "",
        f"数据源：`{session}`；每 50 个 student 更新步记录；各分支 300 个 adapter layer。",
        "",
        "## 当前需要关注",
        "",
    ]
    lines += [f"- {a}" for a in alerts] or ["当前检查未触发告警。"]
    lines += ["", "## 最新分支比较", "", "| 指标（逐层统计） | action | video |", "| --- | ---: | ---: |"]
    for label, key, percent in [
        ("累计 ΔW 范数均值", "delta_w_norm_mean", False),
        ("累计 ΔW / W 均值", "relative_delta_w_norm_mean", True),
        ("累计 ΔW / W 最大值", "relative_delta_w_norm_max", True),
        ("最近 50 步变化 / W 均值", "relative_interval_update_norm_mean", True),
        ("最近 50 步变化 / W 最大值", "relative_interval_update_norm_max", True),
        ("相邻区间更新 cosine 均值", "update_cosine_mean", False),
        ("相邻区间更新 cosine 中位数", "update_cosine_median", False),
        ("cosine < 0 比例", "negative_direction_fraction", True),
        ("cosine < −0.2 比例", "reversal_fraction", True),
        ("未变化比例", "unchanged_fraction", True),
        ("裁剪前 A/B 梯度范数均值", "gradient_norm_mean", False),
        ("非有限权重统计层数", "nonfinite_weight_layer_count", False),
        ("非有限梯度层数", "nonfinite_gradient_layer_count", False),
    ]:
        lines.append(f"| {label} | {fmt(number(latest['action'], key), percent)} | {fmt(number(latest['video'], key), percent)} |")
    lines += [
        "",
        "范数为 Frobenius；方向比例仅以非零且有限的区间更新为分母。两分支 rank/LR 不同：action rank128、LR 1e-4；video rank8、LR 1e-5，不能直接用 A/B 梯度大小比较其作用强弱。",
        "",
        "## 最近记录",
        "",
        "| step | action ΔW/W | video ΔW/W | action cosine | video cosine | action 反转 | video 反转 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in sorted(common_steps)[-10:]:
        pair = {b: next(r for r in summary if r["branch"] == b and int(r["step"]) == s) for b in ("action", "video")}
        cells = [fmt(number(pair[b], key), percent) for key, percent in [("relative_delta_w_norm_mean", True), ("update_cosine_mean", False), ("reversal_fraction", True)] for b in ("action", "video")]
        lines.append("| " + " | ".join([str(s)] + cells) + " |")
    lines += ["", "## Loss 与诊断评测", "", "| step 区间 | 已记录点数 | DMD 均值 | fake 均值 | student grad 均值 | video grad 均值 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for lower in range(1, training_step + 1, 250):
        upper = lower + 249
        selected = [r for r in train if lower <= r["step"] <= upper]
        if selected:
            lines.append("| " + " | ".join([f"{lower}–{min(upper, training_step)}", str(len(selected))] + [fmt(mean(selected, k)) for k in ("dmd", "fake", "student_grad", "video_grad")]) + " |")
    lines += ["", "| eval step | student–teacher L1 | student–GT L1 |", "| ---: | ---: | ---: |"]
    for r in evaluations:
        lines.append(f"| {int(r['step'])} | {fmt(r['student_teacher_l1'])} | {fmt(r['student_gt_l1'])} |")
    lines += [
        "",
        "DMD 数值是随 fake/teacher 信号变化的训练代理目标，不等同于固定验证误差。此配置每 rank 仅 4 个诊断样本；L1 不能替代 RoboTwin 闭环成功率。teacher 也使用当前共享 video condition，因此 student–teacher L1 不是相对原始完整冻结 FastWAM 的固定参照。",
        "",
        "## 层级证据",
        "",
    ]
    for b in ("action", "video"):
        selected = [r for r in layer_rows if r["branch"] == b]
        unchanged = [r["layer"] for r in selected if (number(r, "interval_update_norm") is not None and number(r, "interval_update_norm") <= 1e-12)]
        lines += [f"**{b}：** 最近区间未变化 {len(unchanged)}/{len(selected)} 层。", ""]
        lines += [f"- `{name}`" for name in unchanged]
        lines += ["", "最负的 5 个 cosine：", ""]
        for r in sorted([r for r in selected if number(r, "update_cosine") is not None], key=lambda r: number(r, "update_cosine"))[:5]:
            lines.append(f"- `{r['layer']}`：cos={fmt(number(r, 'update_cosine'))}；区间变化/W={fmt(number(r, 'relative_interval_update_norm'), True)}。")
        lines.append("")
    lines += [
        "video 第 29 block（最后一层）只有 self-attn K/V 向 action 提供缓存；该层 Q、O、cross-attn、FFN 的输出没有后续 video layer 消费。因而目前其余 8 个 adapter 的零更新与计算图结构一致，不表示 video 分支整体未学习。证据：`mot.py:272–313` 的逐层 K/V 缓存与末层输出用途；当前 anchor 在初始化处梯度为零，video weight_decay=0。",
        "",
        "## 解读边界与后续",
        "",
        "video 有效权重非零且持续变化可以证明参与参数学习；持续低反转支持在 50 步时间尺度上相对稳定，不能证明提高任务成功率。action 的负 cosine 提示更新在反复修正，不能据此归因为 action/video 目标冲突。",
        "",
        f"下一诊断节点 step {(training_step // 500 + 1) * 500}，下一保存节点 step {(training_step // 2000 + 1) * 2000}。继续检查 L1 趋势、action 反转和 video 更新尖峰/NaN。比较任务效果需要本轮 checkpoint 的闭环评测和匹配训练预算的 action-only 对照。",
        "",
        "本报告由独立只读观察进程更新，不修改训练配置、权重或 optimizer，不发送外部消息。观察进程停止或当前执行环境消失后不会继续更新；它不会自动向聊天推送通知。",
        "",
    ]
    state = {"updated_at": utc(), "pid": os.getpid(), "training_step": training_step, "monitor_step": step, "log_age_seconds": age, "alerts": alerts, "latest": latest, "evaluations": evaluations}
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "latest.md", "\n".join(lines))
    atomic_text(output / "status.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    # Persist a compact evidence history even after latest.md is refreshed.
    with (output / "observations.jsonl").open("a") as handle:
        handle.write(json.dumps(state, ensure_ascii=False) + "\n")
    print(json.dumps({"time": utc(), "training_step": training_step, "monitor_step": step, "alert_count": len(alerts)}, ensure_ascii=False), flush=True)
    return training_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--until-step", type=int, default=2000)
    parser.add_argument("--max-hours", type=float, default=6)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.max_hours <= 0:
        parser.error("poll-seconds must be >=1 and max-hours must be positive")
    deadline = time.monotonic() + args.max_hours * 3600
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "watch.pid").write_text(str(os.getpid()) + "\n")
    signature = None
    step = 0
    while True:
        try:
            current = tuple(p.stat().st_mtime_ns for p in [args.run_dir / "train.log", args.session_dir / "summary.csv"])
            if current != signature or time.time() - (args.run_dir / "train.log").stat().st_mtime > 900:
                step = inspect(args.run_dir, args.session_dir, args.output_dir)
                signature = current
            reached = args.until_step > 0 and step >= args.until_step
            if reached and args.until_step % 2000 == 0:
                reached = (args.run_dir / f"checkpoint-{args.until_step:09d}" / "training_state.pt").exists()
            if reached and args.until_step == 30000:
                # An old run already has checkpoint-30000 in this directory.
                # Require this run's final log after its save/barrier completes.
                final_log = complete_text(args.run_dir / "train.log")
                final_log = final_log[final_log.rfind("[train] start") :]
                reached = "[train] finished FastWAM action DMD iter=30000" in final_log
            if reached:
                inspect(args.run_dir, args.session_dir, args.output_dir)
            if args.once or reached or time.monotonic() >= deadline:
                break
        except (OSError, ValueError, KeyError, StopIteration) as error:
            print(json.dumps({"time": utc(), "read_error": str(error)}), flush=True)
            if args.once:
                raise
        if time.monotonic() >= deadline:
            break
        time.sleep(args.poll_seconds)
    atomic_text(args.output_dir / "finished.json", json.dumps({"time": utc(), "last_training_step": step, "until_step": args.until_step, "once": args.once}) + "\n")


if __name__ == "__main__":
    main()
