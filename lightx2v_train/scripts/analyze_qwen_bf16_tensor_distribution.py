"""Analyze Qwen-Image BF16 snapshots with the Wan DMD four-scale convention."""

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
from analyze_bf16_tensor_distribution import CODEBOOK, quantize, save_figure, style_axes

PATTERN = re.compile(r"iter_(\d+)_rank_(\d+)_transformer_blocks_(\d+)_(attn_to_[qkv]|img_mlp_(?:proj|out))\.pt$")
TENSORS = {
    "forward_input": "Input",
    "forward_weight": "Weight",
    "forward_output": "Output",
    "backward_grad_output": "Output gradient",
}
FAMILIES = {
    "attn_to_q": "Attention Q",
    "attn_to_k": "Attention K",
    "attn_to_v": "Attention V",
    "img_mlp_proj": "MLP up",
    "img_mlp_out": "MLP down",
}
FAMILY_COLORS = ["#13877f", "#3374a1", "#795b9e", "#c06d37", "#bb4b57"]
SCALES = {"ceil": "ceil", "ceil_minus_1": "ceil - 1", "floor": "floor", "floor_minus_1": "floor - 1"}
SCALE_COLORS = ["#66747d", "#2a8c98", "#d39a35", "#bd5361"]


def analyze(tensor: torch.Tensor, seed: int, sample_blocks: int):
    if tensor.dtype != torch.bfloat16 or tensor.shape[-1] % 16:
        raise ValueError(f"Expected BF16 tensor with last dimension divisible by 16; got {tensor.dtype} {tuple(tensor.shape)}")
    blocks = tensor.reshape(-1, 16)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(blocks), size=min(sample_blocks, len(blocks)), replace=False)
    values = blocks[selected].float().numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Captured tensor contains non-finite values")
    maxima = np.max(np.abs(values), axis=1)
    log_max = np.log2(np.maximum(maxima, np.finfo(np.float32).tiny))
    ceil_exp, floor_exp = np.ceil(log_max), np.floor(log_max)
    exponents = {"ceil": ceil_exp, "ceil_minus_1": ceil_exp - 1, "floor": floor_exp, "floor_minus_1": floor_exp - 1}
    scales = {name: np.exp2(exp - 2) for name, exp in exponents.items()}
    quantized = {name: quantize(values / scale[:, None]) * scale[:, None] for name, scale in scales.items()}
    errors = {name: np.mean((values - output) ** 2, axis=1) for name, output in quantized.items()}
    auto_error = np.minimum(errors["floor"], errors["floor_minus_1"])
    signal = np.sum(values**2)

    def relative(error):
        return float(100 * 16 * np.sum(error) / signal) if signal else 0.0

    nonzero = maxima > 0
    stats = {
        "elements": tensor.numel(),
        "sampled_blocks": len(values),
        "sampled_zero_fraction": float(np.mean(values == 0)),
        "sampled_abs_p99": float(np.quantile(np.abs(values), 0.99)),
        "floor_minus_1_better_fraction": float(np.mean(errors["floor_minus_1"] < errors["floor"])),
        "floor_equals_ceil_minus_1_exponent_nonzero": float(np.mean(floor_exp[nonzero] == ceil_exp[nonzero] - 1)) if np.any(nonzero) else 0.0,
        "floor_equals_ceil_minus_1_block": float(np.mean(np.all(quantized["floor"] == quantized["ceil_minus_1"], axis=1))),
        "floor_equals_ceil_minus_1_element": float(np.mean(quantized["floor"] == quantized["ceil_minus_1"])),
        **{f"relative_mse_{name}_pct": relative(error) for name, error in errors.items()},
        "relative_mse_auto_pct": relative(auto_error),
        "auto_mse_reduction_pct": float(100 * (np.sum(errors["floor"]) - np.sum(auto_error)) / np.sum(errors["floor"])) if np.any(errors["floor"]) else 0.0,
    }
    normalized = (values / scales["ceil"][:, None]).reshape(-1).astype(np.float32)
    return stats, normalized


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_density(samples, output: Path, step: int):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.7), constrained_layout=True)
    bins = np.linspace(-4.2, 4.2, 169)
    for ax, (kind, label) in zip(axes.flat, TENSORS.items()):
        pooled = []
        for (family, family_label), color in zip(FAMILIES.items(), FAMILY_COLORS):
            values = np.concatenate(samples[(kind, family)])
            pooled.append(values)
            counts, edges = np.histogram(values, bins=bins, density=True)
            ax.plot((edges[:-1] + edges[1:]) / 2, np.maximum(counts, 1e-5), color=color, linewidth=1.7, label=family_label)
        combined = np.concatenate(pooled)
        center = np.median(combined)
        width = np.mean(np.abs(combined - center))
        x = np.linspace(-4.2, 4.2, 400)
        ax.plot(x, np.exp(-np.abs(x - center) / width) / (2 * width), color="#737d82", linewidth=1.1, linestyle=(0, (4, 2)), label="Laplace fit")
        ax.vlines(CODEBOOK[np.abs(CODEBOOK) <= 4.2], 1e-5, 2e-5, color="#8c9698", linewidth=0.7)
        ax.set(xlim=(-4.2, 4.2), ylim=(1e-5, 3), yscale="log", xlabel="Value / ceil block scale", ylabel="Probability density")
        ax.set_title(label, loc="left", fontweight="bold", color="#23343a")
        style_axes(ax)
    axes[0, 0].legend(ncol=2, fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle(f"Qwen-Image BF16 distributions  |  iteration {step}", fontsize=15, fontweight="bold", color="#23343a")
    save_figure(fig, output, "qwen_block_normalized_density")


def plot_four_scales(grouped: list[dict], output: Path, step: int):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), constrained_layout=True, sharey=True)
    for ax, (family, label) in zip(axes.flat, FAMILIES.items()):
        for offset, (scale, scale_label), color in zip((-0.27, -0.09, 0.09, 0.27), SCALES.items(), SCALE_COLORS):
            for x, kind in enumerate(TENSORS):
                row = next(r for r in grouped if r["iteration"] == step and r["family"] == family and r["tensor"] == kind)
                key = f"relative_mse_{scale}_pct"
                center, low, high = (row[f"{key}_{suffix}"] for suffix in ("median", "min", "max"))
                ax.errorbar(x + offset, center, yerr=[[center - low], [high - center]], fmt="o", color=color, markersize=4, linewidth=1, capsize=2, label=scale_label if x == 0 else None)
        ax.set(yscale="log", xticks=range(4), xticklabels=["Input", "Weight", "Output", "Grad"], ylabel="Relative MSE (%)")
        ax.set_title(label, loc="left", fontweight="bold", color="#23343a")
        style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles, labels, ncol=2, fontsize=10, frameon=False, loc="center")
    fig.suptitle(f"FP4 E2M1 block-scale error  |  iteration {step}\nMedian and range across 3 blocks x 2 ranks", fontsize=14, fontweight="bold", color="#23343a")
    save_figure(fig, output, "qwen_four_scale_mse")


def plot_early_change(grouped: list[dict], output: Path, steps: list[int]):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4), constrained_layout=True)
    for ax, (kind, label) in zip(axes.flat, TENSORS.items()):
        for step, marker, color in zip(steps, ("o", "s"), ("#3374a1", "#c06d37")):
            values = [next(r["relative_mse_floor_pct_median"] for r in grouped if r["iteration"] == step and r["family"] == family and r["tensor"] == kind) for family in FAMILIES]
            ax.plot(range(len(FAMILIES)), values, marker=marker, color=color, linewidth=1.7, markersize=5, label=f"iter {step}")
        ax.set(xticks=range(len(FAMILIES)), xticklabels=["Q", "K", "V", "Up", "Down"], ylabel="Floor relative MSE (%)")
        ax.set_title(label, loc="left", fontweight="bold", color="#23343a")
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Early training: fixed-floor FP4 error", fontsize=15, fontweight="bold", color="#23343a")
    save_figure(fig, output, "qwen_early_floor_mse")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-blocks", type=int, default=4096)
    args = parser.parse_args()
    if args.sample_blocks < 1:
        parser.error("--sample-blocks must be positive")
    files = sorted(args.capture_dir.glob("iter_*_rank_*_transformer_blocks_*.pt"))
    if not files:
        parser.error("No Qwen BF16 snapshots found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    rows = []
    density_samples = defaultdict(list)
    steps = sorted({int(PATTERN.fullmatch(path.name)[1]) for path in files if PATTERN.fullmatch(path.name)})
    final_step = steps[-1]
    for index, path in enumerate(files, 1):
        match = PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected snapshot name: {path.name}")
        step, rank, block, family = int(match[1]), int(match[2]), int(match[3]), match[4]
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        for kind in TENSORS:
            seed = int.from_bytes(hashlib.sha256(f"{path.name}:{kind}".encode()).digest()[:8], "little")
            stats, normalized = analyze(snapshot[kind], seed, args.sample_blocks)
            rows.append({"iteration": step, "rank": rank, "block": block, "family": family, "tensor": kind, "source_dtype": snapshot["source_dtypes"][list(TENSORS).index(kind)], **stats})
            if step == final_step:
                density_samples[(kind, family)].append(normalized)
        del snapshot
        if index % 10 == 0 or index == len(files):
            print(f"Analyzed {index}/{len(files)} snapshots", flush=True)
    write_csv(args.output_dir / "per_snapshot.csv", rows)
    grouped = []
    for step in steps:
        for family in FAMILIES:
            for kind in TENSORS:
                items = [r for r in rows if r["iteration"] == step and r["family"] == family and r["tensor"] == kind]
                summary = {"iteration": step, "family": family, "tensor": kind, "snapshots": len(items)}
                for key in (
                    "sampled_zero_fraction",
                    "sampled_abs_p99",
                    "floor_minus_1_better_fraction",
                    "floor_equals_ceil_minus_1_exponent_nonzero",
                    "floor_equals_ceil_minus_1_block",
                    "floor_equals_ceil_minus_1_element",
                    "relative_mse_ceil_pct",
                    "relative_mse_ceil_minus_1_pct",
                    "relative_mse_floor_pct",
                    "relative_mse_floor_minus_1_pct",
                    "relative_mse_auto_pct",
                    "auto_mse_reduction_pct",
                ):
                    values = np.array([item[key] for item in items])
                    summary.update({f"{key}_median": float(np.median(values)), f"{key}_min": float(values.min()), f"{key}_max": float(values.max())})
                grouped.append(summary)
    write_csv(args.output_dir / "by_iteration.csv", grouped)
    plot_density(density_samples, args.output_dir, final_step)
    plot_four_scales(grouped, args.output_dir, final_step)
    if len(steps) == 2:
        plot_early_change(grouped, args.output_dir, steps)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "capture_dir": str(args.capture_dir.resolve()),
                "snapshots": len(files),
                "iterations": steps,
                "sample_blocks_per_tensor": args.sample_blocks,
                "block_size": 16,
                "quantization": "nearest FP4 E2M1; scale=2^(exponent-2); Auto chooses floor or floor-1 by block MSE",
                "versions": {"torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"snapshots": len(files), "iterations": steps, "rows": len(rows)}))


if __name__ == "__main__":
    main()
