"""Read-only monitoring of effective LoRA weights, separate from training state."""

import csv
import json
import math
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger


@dataclass(frozen=True)
class LoraMonitorConfig:
    enabled: bool = False
    every_n_steps: int = 50
    reversal_cos_threshold: float = -0.2
    zero_tolerance: float = 1e-12
    save_snapshots: bool = True
    keep_last_snapshots: int = 2

    @classmethod
    def from_mapping(cls, mapping):
        config = cls(**(mapping or {}))
        if isinstance(config.every_n_steps, bool) or not isinstance(config.every_n_steps, int) or config.every_n_steps <= 0:
            raise ValueError("logging.lora_monitor.every_n_steps must be a positive integer")
        if not math.isfinite(config.reversal_cos_threshold) or not -1 <= config.reversal_cos_threshold < 0:
            raise ValueError("logging.lora_monitor.reversal_cos_threshold must be in [-1, 0)")
        if not math.isfinite(config.zero_tolerance) or config.zero_tolerance < 0:
            raise ValueError("logging.lora_monitor.zero_tolerance must be finite and non-negative")
        if isinstance(config.keep_last_snapshots, bool) or not isinstance(config.keep_last_snapshots, int) or config.keep_last_snapshots < 0:
            raise ValueError("logging.lora_monitor.keep_last_snapshots must be a non-negative integer (0 keeps all)")
        return config


def _product_inner(left, right):
    """<L R, U V>_F without materializing the dense weight matrices."""
    l, r = left
    u, v = right
    return ((l.T @ u) * (r @ v.T)).sum()


def _product_norm(product):
    return float(_product_inner(product, product).clamp_min(0).sqrt())


def _factors(snapshot, device):
    return snapshot["b"].to(device=device, dtype=torch.float64) * snapshot["scale"], snapshot["a"].to(device=device, dtype=torch.float64)


def _difference(current, previous):
    # B A - D C = B (A-C) + (B-D) C. Subtract factors first to avoid
    # subtracting nearly equal squared norms for small optimizer updates.
    b, a = current
    d, c = previous
    return torch.cat((b, b - d), dim=1), torch.cat((a - c, c), dim=0)


def _finite(value):
    return value if value is not None and math.isfinite(value) else None


class LoraChangeMonitor:
    """Call only on global rank zero, after distributed parameter synchronization.

    Norms are flattened L2 / Frobenius, not matrix spectral norms. History is
    stored as CPU factors; statistics use FP64 low-rank Gram products.
    """

    measures = ("delta_w_norm", "relative_delta_w_norm", "interval_update_norm", "relative_interval_update_norm", "update_cosine", "gradient_norm", "reversal_fraction_so_far")

    @torch.no_grad()
    def __init__(self, config, branches, output_dir, initial_step=0):
        self.config = config
        self.initial_step = initial_step
        self.last_step = initial_step
        self.layers = {branch: [] for branch in branches}
        self.history = {}
        self.gradients = {}
        self.direction_counts = {}
        self.directory = Path(output_dir) / "lora_monitor" / f"from-{initial_step:09d}-{uuid.uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        for branch, expert in branches.items():
            if expert is None:
                continue
            for name, layer in expert.named_modules():
                if not hasattr(layer, "lora_A") or not hasattr(layer, "lora_B"):
                    continue
                for adapter in layer.lora_A:
                    a, b = layer.lora_A[adapter].weight, layer.lora_B[adapter].weight
                    if a.ndim != 2 or b.ndim != 2 or getattr(layer, "use_dora", {}).get(adapter, False):
                        raise ValueError("LoRA monitoring supports standard linear BA adapters only (not convolution or DoRA)")
                    if getattr(layer.lora_B[adapter], "bias", None) is not None:
                        raise ValueError("LoRA monitoring requires bias-free LoRA B projections")
                    key = f"{branch}/{name}/{adapter}"
                    with torch.autocast(device_type=a.device.type, enabled=False):
                        base_norm = float(torch.linalg.vector_norm(layer.get_base_layer().weight.detach(), dtype=torch.float64))
                    self.layers[branch].append((key, name, adapter, layer, base_norm))
                    self.direction_counts[key] = [0, 0]
        metadata = {
            "config": asdict(config),
            "initial_step": initial_step,
            "norm": "flattened L2 (Frobenius)",
            "gradient": "accumulated and synchronized, before clipping, sampled update only",
            "cosine": "between consecutive recorded interval updates of scale*B*A",
            "resume": "new session baseline; no direction comparisons across process restarts",
            "layers": {branch: [entry[0] for entry in layers] for branch, layers in self.layers.items()},
        }
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.record(initial_step, baseline=True)
        logger.info("[lora-monitor] enabled every={} steps, CSV/snapshots={}", config.every_n_steps, self.directory)

    def due(self, step):
        return step > self.initial_step and step % self.config.every_n_steps == 0

    @torch.no_grad()
    def capture_gradients(self, branch, step):
        if not self.due(step):
            return
        for key, _, adapter, layer, _ in self.layers[branch]:
            parameters = [layer.lora_A[adapter].weight, layer.lora_B[adapter].weight]
            present = [p.grad.detach() for p in parameters if p.grad is not None]
            with torch.autocast(device_type=parameters[0].device.type, enabled=False):
                norm = math.sqrt(sum(float(torch.linalg.vector_norm(g, dtype=torch.float64)) ** 2 for g in present))
            self.gradients[key] = (step, norm, len(present), sum(p.requires_grad for p in parameters))

    @staticmethod
    def _snapshot(layer, adapter):
        active = adapter in layer.active_adapters and not layer.disable_adapters
        return {
            "a": layer.lora_A[adapter].weight.detach().to(device="cpu", dtype=torch.float32).clone(),
            "b": layer.lora_B[adapter].weight.detach().to(device="cpu", dtype=torch.float32).clone(),
            "scale": float(layer.scaling[adapter]) if active else 0.0,
            "active": active,
        }

    @staticmethod
    def _append_csv(path, rows):
        if not rows:
            return
        exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    @torch.no_grad()
    def record(self, step, *, baseline=False):
        if not baseline and (not self.due(step) or step <= self.last_step):
            return {}
        started = time.perf_counter()
        rows, summaries, scalars, snapshots = [], [], {}, {}
        for branch, layers in self.layers.items():
            branch_rows = []
            for key, name, adapter, layer, base_norm in layers:
                current = self._snapshot(layer, adapter)
                previous = self.history.get(key, [])
                device = layer.lora_A[adapter].weight.device
                change_norm, cosine = None, None
                with torch.autocast(device_type=device.type, enabled=False):
                    product = _factors(current, device)
                    norm = _product_norm(product)
                    if previous:
                        last = _factors(previous[-1], device)
                        change = _difference(product, last)
                        change_norm = _product_norm(change)
                        if len(previous) == 2:
                            before = _factors(previous[0], device)
                            old_change = _difference(last, before)
                            old_norm = _product_norm(old_change)
                            if math.isfinite(change_norm) and math.isfinite(old_norm) and min(change_norm, old_norm) > self.config.zero_tolerance:
                                value = float(_product_inner(change, old_change)) / (change_norm * old_norm)
                                if math.isfinite(value):
                                    cosine = max(-1.0, min(1.0, value))
                counts = self.direction_counts[key]
                reversed_direction = None if cosine is None else int(cosine < self.config.reversal_cos_threshold)
                if cosine is not None:
                    counts[0] += 1
                    counts[1] += reversed_direction
                gradient = self.gradients.get(key)
                captured = gradient is not None and gradient[0] == step and not baseline
                row = {
                    "step": step,
                    "previous_step": None if baseline else self.last_step,
                    "branch": branch,
                    "layer": name,
                    "adapter": adapter,
                    "active": int(current["active"]),
                    "trainable": int(layer.lora_A[adapter].weight.requires_grad or layer.lora_B[adapter].weight.requires_grad),
                    "base_weight_norm": _finite(base_norm),
                    "delta_w_norm": _finite(norm),
                    "relative_delta_w_norm": _finite(norm / base_norm) if base_norm > 0 else None,
                    "interval_update_norm": _finite(change_norm),
                    "relative_interval_update_norm": _finite(change_norm / base_norm) if base_norm > 0 and change_norm is not None else None,
                    "update_cosine": cosine,
                    "negative_direction": None if cosine is None else int(cosine < 0),
                    "reversed_direction": reversed_direction,
                    "direction_observations": counts[0],
                    "reversal_fraction_so_far": counts[1] / counts[0] if counts[0] else None,
                    "gradient_norm": _finite(gradient[1]) if captured else None,
                    "gradient_tensors_present": gradient[2] if captured else None,
                    "gradient_tensors_trainable": gradient[3] if captured else None,
                    "nonfinite_gradient": int(not math.isfinite(gradient[1])) if captured else None,
                    "nonfinite_weight_stat": int(not math.isfinite(norm) or (change_norm is not None and not math.isfinite(change_norm))),
                }
                branch_rows.append(row)
                self.history[key] = (previous + [current])[-2:]
                snapshots[key] = current
            rows.extend(branch_rows)
            valid_directions = [r for r in branch_rows if r["update_cosine"] is not None]
            changes = [r["interval_update_norm"] for r in branch_rows if r["interval_update_norm"] is not None]
            grads = [r for r in branch_rows if r["gradient_tensors_present"] is not None]
            summary = {
                "step": step,
                "branch": branch,
                "layer_count": len(layers),
                "has_lora": int(bool(layers)),
                "trainable_layer_count": sum(r["trainable"] for r in branch_rows),
                "valid_direction_count": len(valid_directions),
                "negative_direction_fraction": sum(r["negative_direction"] for r in valid_directions) / len(valid_directions) if valid_directions else None,
                "reversal_fraction": sum(r["reversed_direction"] for r in valid_directions) / len(valid_directions) if valid_directions else None,
                "unchanged_fraction": sum(v <= self.config.zero_tolerance for v in changes) / len(changes) if changes else None,
                "missing_gradient_fraction": sum(r["gradient_tensors_present"] == 0 for r in grads) / len(grads) if grads else None,
                "zero_gradient_fraction": sum(r["gradient_norm"] == 0 for r in grads) / len(grads) if grads else None,
                "nonfinite_weight_layer_count": sum(r["nonfinite_weight_stat"] for r in branch_rows),
                "nonfinite_gradient_layer_count": sum(r["nonfinite_gradient"] or 0 for r in branch_rows),
            }
            for measure in self.measures:
                values = [r[measure] for r in branch_rows if r[measure] is not None]
                summary.update(
                    {
                        f"{measure}_mean": statistics.fmean(values) if values else None,
                        f"{measure}_median": statistics.median(values) if values else None,
                        f"{measure}_max": max(values) if values else None,
                        f"{measure}_count": len(values),
                    }
                )
            summaries.append(summary)
            scalars.update({f"lora_monitor/{branch}/{key}": value for key, value in summary.items() if key not in {"step", "branch"} and value is not None})
            logger.info(
                "[lora-monitor] step={} branch={} layers={} delta_mean={} update_mean={} grad_mean={} cosine_mean={} reversed={}",
                step,
                branch,
                len(layers),
                summary["delta_w_norm_mean"],
                summary["interval_update_norm_mean"],
                summary["gradient_norm_mean"],
                summary["update_cosine_mean"],
                summary["reversal_fraction"],
            )
        self._append_csv(self.directory / "layers.csv", rows)
        self._append_csv(self.directory / "summary.csv", summaries)
        if self.config.save_snapshots:
            path = self.directory / f"step-{step:09d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save({"step": step, "layers": snapshots, "direction_counts": self.direction_counts}, temporary)
            temporary.replace(path)
            if self.config.keep_last_snapshots:
                for old in sorted(self.directory.glob("step-*.pt"))[: -self.config.keep_last_snapshots]:
                    old.unlink()
        self.last_step = step
        self.gradients.clear()
        scalars["lora_monitor/seconds"] = time.perf_counter() - started
        return scalars
