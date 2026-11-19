"""Analyze Qwen-Image BF16 snapshots with the MXFP4 `_quantize_mx` expshift sweep.

Mirrors the Wan "mxfp4量化分析" convention: for each 32-element block, quantize with
``fake_quant_ops.quant.ops.mxfp._quantize_mx`` (scale_bits=8, fp4_e2m1) at nine
``minus_exp`` shifts and report per-block MSE, the average MSE-vs-expshift per module,
and the per-block optimal shift (0 / -1) count and ratio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fake_quant_ops.quant.ops.mxfp import _quantize_mx

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
TENSOR_MARKERS = {"forward_input": "o", "forward_weight": "s", "forward_output": "^", "backward_grad_output": "D"}
TENSOR_COLORS = {"forward_input": "#2a8c98", "forward_weight": "#d39a35", "forward_output": "#3374a1", "backward_grad_output": "#bd5361"}
MINUS_EXPS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def quantize_blocks(blocks: torch.Tensor, minus_exp: int) -> torch.Tensor:
    """Quantize a (num_blocks, 32) float32 tensor with one shared E8M0 scale per block."""
    return _quantize_mx(
        blocks,
        scale_bits=8,
        elem_format="fp4_e2m1",
        shared_exp_method="max",
        axes=-1,
        block_size=32,
        round="nearest",
        flush_fp32_subnorms=False,
        minus_exp=minus_exp,
    )


def analyze(tensor: torch.Tensor, seed: int, sample_blocks: int) -> dict:
    if tensor.dtype != torch.bfloat16 or tensor.shape[-1] % 32:
        raise ValueError(f"Expected BF16 tensor with last dimension divisible by 32; got {tensor.dtype} {tuple(tensor.shape)}")
    blocks = tensor.reshape(-1, 32)
    rng = np.random.default_rng(seed)
    selected = blocks[rng.choice(len(blocks), size=min(sample_blocks, len(blocks)), replace=False)].float()
    if not torch.isfinite(selected).all():
        raise ValueError("Captured tensor contains non-finite values")

    signal = float(torch.sum(selected**2))
    mse_by_shift = {}
    block_mse_by_shift = {}
    for minus_exp in MINUS_EXPS:
        quantized = quantize_blocks(selected, minus_exp)
        block_mse = torch.mean((selected - quantized) ** 2, dim=1)
        block_mse_by_shift[minus_exp] = block_mse
        mse_by_shift[minus_exp] = float(100 * 32 * torch.sum(block_mse) / signal) if signal else 0.0

    stacked = torch.stack([block_mse_by_shift[minus_exp] for minus_exp in MINUS_EXPS], dim=1)
    optimal = torch.argmin(stacked, dim=1)
    optimal_shift = torch.tensor([MINUS_EXPS[index] for index in optimal.tolist()])
    block_max = selected.abs().amax(dim=1)
    valid = block_max > 0
    zero_blocks = int((~valid).sum())
    valid_blocks = int(valid.sum())
    nonzero = selected.reshape(-1).abs() > 0

    stats = {
        "elements": tensor.numel(),
        "sampled_blocks": len(selected),
        **{f"relative_mse_shift_{minus_exp:+d}_pct": mse_by_shift[minus_exp] for minus_exp in MINUS_EXPS},
        "valid_blocks": valid_blocks,
        "zero_blocks": zero_blocks,
        "optimal_shift_0_blocks": int(((optimal_shift == 0) & valid).sum()),
        "optimal_shift_minus_1_blocks": int(((optimal_shift == -1) & valid).sum()),
        "optimal_shift_0_fraction": float(((optimal_shift == 0) & valid).sum() / valid_blocks) if valid_blocks else 0.0,
        "optimal_shift_minus_1_fraction": float(((optimal_shift == -1) & valid).sum() / valid_blocks) if valid_blocks else 0.0,
        "nonzero_fraction": float(nonzero.float().mean()),
    }
    return stats


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


def plot_mse_expshift(grouped: list[dict], output: Path, step: int):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)
    for ax, (family, family_label) in zip(axes.flat, FAMILIES.items()):
        for kind, label in TENSORS.items():
            rows = [r for r in grouped if r["iteration"] == step and r["family"] == family and r["tensor"] == kind]
            if not rows:
                continue
            row = rows[0]
            y = np.array([row[f"relative_mse_shift_{minus_exp:+d}_pct_median"] for minus_exp in MINUS_EXPS])
            ax.plot(MINUS_EXPS, y, marker=TENSOR_MARKERS[kind], markersize=4, linewidth=1.7, color=TENSOR_COLORS[kind], label=label)
        ax.set(yscale="log", xticks=MINUS_EXPS, xlabel="minus_exp (expshift)", ylabel="Relative MSE (%)")
        ax.set_title(family_label, loc="left", fontweight="bold", color="#23343a")
        style_axes(ax)
    axes.flat[-1].axis("off")
    handles = [plt.Line2D([], [], color=TENSOR_COLORS[kind], marker=TENSOR_MARKERS[kind], markersize=5, linewidth=1.7, label=label) for kind, label in TENSORS.items()]
    axes.flat[-1].legend(handles=handles, frameon=False, loc="center", fontsize=10)
    fig.suptitle(f"MXFP4 E2M1 block-scale error vs expshift  |  iteration {step}", fontsize=14, fontweight="bold", color="#23343a")
    save_figure(fig, output, "qwen_mxfp4_expshift_mse")


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
    steps = sorted({int(PATTERN.fullmatch(path.name)[1]) for path in files if PATTERN.fullmatch(path.name)})
    for index, path in enumerate(files, 1):
        match = PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected snapshot name: {path.name}")
        step, rank, block, family = int(match[1]), int(match[2]), int(match[3]), match[4]
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        for kind in TENSORS:
            seed = int.from_bytes(hashlib.sha256(f"{path.name}:{kind}".encode()).digest()[:8], "little")
            stats = analyze(snapshot[kind], seed, args.sample_blocks)
            rows.append({"iteration": step, "rank": rank, "block": block, "family": family, "tensor": kind, **stats})
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
                for key in [f"relative_mse_shift_{minus_exp:+d}_pct" for minus_exp in MINUS_EXPS] + [
                    "optimal_shift_0_fraction",
                    "optimal_shift_minus_1_fraction",
                    "optimal_shift_0_blocks",
                    "optimal_shift_minus_1_blocks",
                    "nonzero_fraction",
                ]:
                    values = np.array([item[key] for item in items])
                    summary[f"{key}_median"] = float(np.median(values))
                grouped.append(summary)
    write_csv(args.output_dir / "by_family_tensor.csv", grouped)

    optimal_rows = []
    for family in FAMILIES:
        for kind in TENSORS:
            items = [r for r in rows if r["family"] == family and r["tensor"] == kind]
            total = sum(r["sampled_blocks"] for r in items)
            zeros = sum(r["optimal_shift_0_blocks"] for r in items)
            minus_ones = sum(r["optimal_shift_minus_1_blocks"] for r in items)
            optimal_rows.append(
                {
                    "family": family,
                    "tensor": kind,
                    "total_blocks": total,
                    "valid_blocks": sum(r["valid_blocks"] for r in items),
                    "zero_blocks": sum(r["zero_blocks"] for r in items),
                    "shift_0_blocks": zeros,
                    "shift_0_ratio": zeros / sum(r["valid_blocks"] for r in items) if sum(r["valid_blocks"] for r in items) else 0.0,
                    "shift_minus_1_blocks": minus_ones,
                    "shift_minus_1_ratio": minus_ones / sum(r["valid_blocks"] for r in items) if sum(r["valid_blocks"] for r in items) else 0.0,
                }
            )
    write_csv(args.output_dir / "optimal_shift.csv", optimal_rows)

    final_step = steps[-1]
    plot_mse_expshift(grouped, args.output_dir, final_step)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "capture_dir": str(args.capture_dir.resolve()),
                "snapshots": len(files),
                "iterations": steps,
                "sample_blocks_per_tensor": args.sample_blocks,
                "block_size": 32,
                "scale_bits": 8,
                "quantization": "fake_quant_ops _quantize_mx fp4_e2m1; nine minus_exp shifts [-4..4]",
                "versions": {"torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"snapshots": len(files), "iterations": steps, "rows": len(rows)}))


if __name__ == "__main__":
    main()
