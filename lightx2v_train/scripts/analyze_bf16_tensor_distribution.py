"""Analyze captured BF16 DMD tensors and render reproducible distribution figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PATTERN = re.compile(r"iter_(\d+)_rank_(\d+)_blocks_(\d+)_(self_attn_q|ffn_0)\.pt$")
TENSORS = {
    "forward_input": "Input",
    "forward_weight": "Weight",
    "forward_output": "Output",
    "backward_grad_output": "Output gradient",
}
FAMILIES = {"self_attn_q": "Attention Q", "ffn_0": "FFN up"}
COLORS = {"self_attn_q": "#087e8b", "ffn_0": "#cf593f"}
SCALE_COLORS = {"ceil": "#50616d", "ceil_minus_1": "#188491", "floor": "#d29335", "floor_minus_1": "#c95748"}
SCALE_LABELS = {"ceil": "ceil", "ceil_minus_1": "ceil - 1", "floor": "floor", "floor_minus_1": "floor - 1"}
CODEBOOK = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
BINS = np.linspace(-8, 8, 161)


def quantize(values: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(CODEBOOK, values), 1, len(CODEBOOK) - 1)
    left = right - 1
    left_distance = values - CODEBOOK[left]
    right_distance = CODEBOOK[right] - values
    choose_left = (left_distance < right_distance) | ((left_distance == right_distance) & (values < 0))
    return np.where(choose_left, CODEBOOK[left], CODEBOOK[right])


def analyze_tensor(tensor: torch.Tensor, seed: int, sample_blocks: int):
    values = tensor.float().numpy().reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("captured tensor contains non-finite values")
    if values.size % 16:
        raise ValueError("last-dimension blocks must contain 16 values")
    rng = np.random.default_rng(seed)
    blocks = values.reshape(-1, 16)
    selected = blocks[rng.integers(0, len(blocks), size=min(sample_blocks, len(blocks)))].astype(np.float64)
    max_abs = np.max(np.abs(selected), axis=1)
    log_max = np.log2(np.maximum(max_abs, np.finfo(np.float32).tiny))
    ceil_exp, floor_exp = np.ceil(log_max), np.floor(log_max)
    exponents = {"ceil": ceil_exp, "ceil_minus_1": ceil_exp - 1,
                 "floor": floor_exp, "floor_minus_1": floor_exp - 1}
    scales = {name: np.exp2(exp - 2) for name, exp in exponents.items()}
    quantized = {name: quantize(selected / scale[:, None]) * scale[:, None]
                 for name, scale in scales.items()}
    mse = {name: np.mean((selected - values) ** 2, axis=1)
           for name, values in quantized.items()}
    mse_auto = np.minimum(mse["ceil"], mse["ceil_minus_1"])
    normalized = selected / scales["ceil"][:, None]
    signal = np.sum(selected**2)
    sampled = selected.reshape(-1)
    abs_sample = np.abs(sampled)
    stats = {
        "elements": values.size,
        "sampled_blocks": len(selected),
        "zero_fraction": float(np.count_nonzero(values == 0) / values.size),
        "mean": float(np.mean(values, dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))),
        "max_abs": float(np.max(np.abs(values))),
        "p50_abs": float(np.quantile(abs_sample, 0.5)),
        "p99_abs": float(np.quantile(abs_sample, 0.99)),
        "p999_abs": float(np.quantile(abs_sample, 0.999)),
        "half_better_fraction": float(np.mean(mse["ceil_minus_1"] < mse["ceil"])),
        "floor_matches_full_fraction": float(np.mean(floor_exp == ceil_exp)),
        "floor_matches_ceil_minus_1_fraction": float(np.mean(floor_exp == ceil_exp - 1)),
        "floor_ceil_minus_1_output_block_agreement": float(np.mean(np.all(quantized["floor"] == quantized["ceil_minus_1"], axis=1))),
        "floor_ceil_minus_1_output_element_agreement": float(np.mean(quantized["floor"] == quantized["ceil_minus_1"])),
        "half_clip_fraction": float(np.mean(np.abs(normalized) > 3)),
        "full_clip_fraction": float(np.mean(np.abs(normalized) > 6)),
        **{f"mse_{name}_over_signal": float(16 * np.sum(errors) / signal) if signal else 0.0
           for name, errors in mse.items()},
        "mse_auto_over_signal": float(16 * np.sum(mse_auto) / signal) if signal else 0.0,
        "auto_mse_reduction": float(np.sum(mse["floor"] - mse_auto) / np.sum(mse["floor"])) if np.any(mse["floor"]) else 0.0,
    }
    return stats, normalized.reshape(-1).astype(np.float32)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aab3ba")
    ax.tick_params(colors="#34404a", labelsize=9)
    ax.grid(axis="y", color="#dce2e5", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, output: Path, name: str):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def density_figure(final_samples: dict, output: Path, final_step: int):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0), constrained_layout=True)
    for ax, (kind, title) in zip(axes.flat, TENSORS.items()):
        pooled = []
        for family in FAMILIES:
            samples = np.concatenate(final_samples[(kind, family)])
            pooled.append(samples)
            counts, edges = np.histogram(samples, bins=BINS, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, np.maximum(counts, 1e-5), color=COLORS[family], linewidth=1.9, label=FAMILIES[family])
        combined = np.concatenate(pooled)
        median = np.median(combined)
        laplace_scale = np.mean(np.abs(combined - median))
        x = np.linspace(-8, 8, 400)
        fit = np.exp(-np.abs(x - median) / laplace_scale) / (2 * laplace_scale)
        ax.plot(x, fit, color="#66727b", linestyle=(0, (4, 2)), linewidth=1.2, label="Laplace fit")
        ax.vlines(CODEBOOK, 1.1e-5, 2.0e-5, color="#69727a", linewidth=1.0, label="ceil levels")
        ax.vlines(CODEBOOK / 2, 2.1e-5, 3.6e-5, color="#b8872c", linewidth=1.0, label="ceil - 1 levels")
        ax.set(xlim=(-8, 8), ylim=(1e-5, 2), yscale="log", xlabel="Value / ceil block scale", ylabel="Probability density")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#27343c")
        style_axes(ax)
    axes[0, 0].legend(ncol=2, fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"BF16 values in FP4 block coordinates  |  iteration {final_step:,}", fontsize=14, fontweight="bold", color="#22313a")
    save_figure(fig, output, "block_normalized_density")


def trend_figure(grouped: list[dict], output: Path, metric: str, name: str, title: str, log_y: bool):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0), constrained_layout=True)
    for ax, (kind, label) in zip(axes.flat, TENSORS.items()):
        for family in FAMILIES:
            rows = [r for r in grouped if r["tensor"] == kind and r["family"] == family]
            rows.sort(key=lambda r: r["iteration"])
            x = np.array([r["iteration"] for r in rows])
            y = np.array([r[f"{metric}_median"] for r in rows])
            lo = np.array([r[f"{metric}_min"] for r in rows])
            hi = np.array([r[f"{metric}_max"] for r in rows])
            color = COLORS[family]
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=color, linewidth=1.8, label=FAMILIES[family])
        ax.set(xscale="log", xlabel="Training iteration", ylabel="Absolute tensor value" if log_y else "Fraction of blocks")
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.75, 0.9)
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color="#27343c")
        ax.set_xticks([1, 10, 100, 1000, 5000], labels=["1", "10", "100", "1k", "5k"])
        style_axes(ax)
    axes[0, 0].legend(ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#22313a")
    save_figure(fig, output, name)


def four_scale_figure(grouped: list[dict], output: Path, final_step: int):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), constrained_layout=True, sharex=True, sharey=True)
    for ax, (family, family_label) in zip(axes, FAMILIES.items()):
        for offset, name in zip((-0.27, -0.09, 0.09, 0.27), SCALE_LABELS):
            for col, kind in enumerate(TENSORS):
                item = next(r for r in grouped if r["iteration"] == final_step and r["family"] == family and r["tensor"] == kind)
                key = f"mse_{name}_over_signal"
                median = 100 * item[f"{key}_median"]
                low = 100 * item[f"{key}_min"]
                high = 100 * item[f"{key}_max"]
                ax.errorbar(col + offset, median, yerr=[[median - low], [high - median]],
                            fmt="o", markersize=5, color=SCALE_COLORS[name], capsize=2,
                            linewidth=1.1, label=SCALE_LABELS[name] if col == 0 else None)
        ax.set(yscale="log", ylim=(0.9, 22), ylabel="Relative MSE (%)")
        ax.set_yticks([1, 2, 5, 10, 20], labels=["1", "2", "5", "10", "20"])
        ax.set_title(family_label, loc="left", fontsize=11, fontweight="bold", color="#27343c")
        style_axes(ax)
    axes[0].legend(ncol=4, frameon=False, loc="upper center", fontsize=9)
    axes[1].set_xticks(range(4), TENSORS.values())
    fig.suptitle(f"FP4 E2M1 error by block exponent  |  iteration {final_step:,}\nDots: median; whiskers: range over 3 blocks x 2 ranks", fontsize=13, fontweight="bold", color="#22313a")
    save_figure(fig, output, "four_scale_mse_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-blocks", type=int, default=4096)
    args = parser.parse_args()
    if args.sample_blocks < 1:
        parser.error("--sample-blocks must be positive")
    files = sorted(args.capture_dir.glob("iter_*_rank_*_blocks_*.pt"))
    if not files:
        parser.error("no tensor snapshots found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.titlesize": 12})
    parsed = []
    for path in files:
        match = PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"unexpected tensor snapshot: {path.name}")
        parsed.append((path, int(match[1]), int(match[2]), int(match[3]), match[4]))
    final_step = max(item[1] for item in parsed)
    rows = []
    final_samples = defaultdict(list)
    for index, (path, iteration, rank, block, family) in enumerate(parsed, 1):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        for kind in TENSORS:
            seed = int.from_bytes(hashlib.sha256(f"{path.name}:{kind}".encode()).digest()[:8], "little")
            stats, normalized = analyze_tensor(snapshot[kind], seed, args.sample_blocks)
            rows.append({"iteration": iteration, "rank": rank, "block": block, "family": family, "tensor": kind, **stats})
            if iteration == final_step:
                final_samples[(kind, family)].append(normalized)
        if index % 24 == 0 or index == len(parsed):
            print(f"analyzed {index}/{len(parsed)} snapshots", flush=True)
    write_csv(args.output_dir / "per_snapshot.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["iteration"], row["family"], row["tensor"])].append(row)
    grouped = []
    for (iteration, family, kind), items in sorted(groups.items()):
        summary = {"iteration": iteration, "family": family, "tensor": kind, "snapshots": len(items)}
        for metric in ("p99_abs", "zero_fraction", "half_better_fraction", "floor_matches_full_fraction", "floor_matches_ceil_minus_1_fraction", "floor_ceil_minus_1_output_block_agreement", "floor_ceil_minus_1_output_element_agreement", "half_clip_fraction", "mse_ceil_over_signal", "mse_ceil_minus_1_over_signal", "mse_floor_over_signal", "mse_floor_minus_1_over_signal", "mse_auto_over_signal", "auto_mse_reduction"):
            values = np.array([item[metric] for item in items])
            summary.update({f"{metric}_median": float(np.median(values)), f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max())})
        grouped.append(summary)
    write_csv(args.output_dir / "by_iteration.csv", grouped)
    density_figure(final_samples, args.output_dir, final_step)
    trend_figure(grouped, args.output_dir, "p99_abs", "p99_over_training", "Tensor magnitude across DMD training  |  median and range over 3 blocks, 2 ranks", True)
    trend_figure(grouped, args.output_dir, "half_better_fraction", "half_scale_preference", "Blocks favoring S/2 by FP4 MSE  |  median and range over 3 blocks, 2 ranks", False)
    four_scale_figure(grouped, args.output_dir, final_step)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "capture_dir": str(args.capture_dir.resolve()), "snapshots": len(files), "iterations": sorted({item[1] for item in parsed}),
        "sample_blocks_per_tensor": args.sample_blocks, "block_size": 16, "quantization": "nearest FP4 E2M1 codebook; four exponents ceil, ceil-1, floor, floor-1 of log2(block max); scale=2^(exponent-2)",
        "versions": {"python_torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
