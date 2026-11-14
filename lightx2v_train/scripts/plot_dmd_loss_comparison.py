"""Plot rank-0 DMD and fake training losses for the six quantization runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = (
    ("Pure DMD", "wan2_1_t2v_1_3b_dmd_vigen", "#38434a"),
    ("NVFP4 F", "wan2_1_t2v_1_3b_dmd_qat", "#087e8b"),
    ("NVFP4 F+B", "wan2_1_t2v_1_3b_dmd_qat_bwd_nvfp4", "#2864ae"),
    ("MXFP4 F (floor)", "wan2_1_t2v_1_3b_dmd_qat_mxfp4_forward_mtc_test", "#c48422"),
    ("MXFP4 F+B (floor)", "wan2_1_t2v_1_3b_dmd_qat_mxfp4", "#c64f40"),
    ("MXFP4 Auto F+B", "wan2_1_t2v_1_3b_dmd_qat_mxfp4_auto", "#8052a2"),
)
LINE = re.compile(r"\[train\] iter=(\d+)/(\d+) dmd=([^ ]+).*? fake=([^ ]+)")


def read_logs(root: Path):
    runs = []
    rows = []
    for label, directory, color in RUNS:
        log = root / directory / "train.log"
        values = []
        rewinds = 0
        for line in log.read_text(errors="replace").splitlines():
            match = LINE.search(line)
            if match:
                iteration, planned = int(match[1]), int(match[2])
                dmd, fake = float(match[3]), float(match[4])
                if not (math.isfinite(dmd) and math.isfinite(fake)):
                    raise ValueError(f"non-finite loss in {log} at iteration {iteration}")
                if values and iteration <= values[-1][0]:
                    values = [row for row in values if row[0] < iteration]
                    rewinds += 1
                values.append((iteration, planned, dmd, fake))
        if not values or [row[0] for row in values] != list(range(1, values[-1][0] + 1)):
            raise ValueError(f"missing or duplicate iterations in {log}")
        if len({row[1] for row in values}) != 1:
            raise ValueError(f"inconsistent planned iteration count in {log}")
        runs.append({"label": label, "directory": directory, "color": color, "planned": values[0][1], "last_iteration": values[-1][0], "rewinds": rewinds, "values": values})
        rows.extend({"run": label, "iteration": iteration, "dmd_loss": dmd, "fake_loss": fake} for iteration, _, dmd, fake in values)
    return runs, rows


def rolling_median(values: np.ndarray, window: int = 25):
    return np.array([np.median(values[max(0, i - window + 1):i + 1]) for i in range(len(values))])


def plot_metric(runs: list[dict], metric: str, output: Path):
    title, ylabel, filename, index = (
        ("DMD loss across quantization conditions", "DMD loss", "dmd_loss_comparison", 2)
        if metric == "dmd" else
        ("Fake loss across quantization conditions", "Fake loss", "fake_loss_comparison", 3)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": (1.7, 1)})
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.70, wspace=0.22)
    maximum = max(row[index] for run in runs for row in run["values"])
    handles = []
    for ax, limit, panel in zip(axes, (1000, 200), ("Full observed history", "First 200 iterations")):
        for run in runs:
            values = np.array(run["values"])
            steps = values[:, 0]
            raw = values[:, index]
            smoothed = rolling_median(raw)
            ax.plot(steps, raw, color=run["color"], linewidth=0.7, alpha=0.14, rasterized=True)
            line, = ax.plot(steps, smoothed, color=run["color"], linewidth=2.0, label=run["label"])
            if ax is axes[0]:
                ax.scatter([steps[-1]], [smoothed[-1]], s=15, color=run["color"], zorder=5)
                handles.append(line)
        ax.set(xlim=(0, limit), ylim=(0, maximum * 1.05), xlabel="Training iteration")
        ax.set_title(panel, loc="left", fontsize=11, color="#2b3942", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#aeb8bd")
        ax.tick_params(labelsize=9, colors="#40505a")
        ax.grid(axis="y", color="#e1e6e8", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.07, y=0.96, ha="left", fontsize=15, fontweight="bold", color="#20313b")
    legend_order = (0, 3, 1, 4, 2, 5)
    fig.legend([handles[i] for i in legend_order], [runs[i]["label"] for i in legend_order], ncol=3, loc="upper left", bbox_to_anchor=(0.07, 0.90), frameon=False, fontsize=9, columnspacing=1.6, handlelength=2.4)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output / f"{filename}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_output_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    runs, rows = read_logs(args.train_output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    with (args.output_dir / "loss_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run", "iteration", "dmd_loss", "fake_loss"))
        writer.writeheader()
        writer.writerows(rows)
    plot_metric(runs, "dmd", args.output_dir)
    plot_metric(runs, "fake", args.output_dir)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "rank-0 train.log [train] iter=... dmd=... fake=...",
        "smoothing": "trailing 25-iteration median; faint lines are unsmoothed values",
        "runs": [{key: run[key] for key in ("label", "directory", "planned", "last_iteration", "rewinds")} for run in runs],
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }, indent=2), encoding="utf-8")
    print("plotted", len(rows), "loss points from", len(runs), "runs")


if __name__ == "__main__":
    main()
