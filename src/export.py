"""
Export: Write evaluation results to CSV, JSON, or formatted console table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from tabulate import tabulate

from .evaluator import EvalResult, LayerResult


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def format_layer_table(result: EvalResult, top_n: int | None = None) -> str:
    """
    Format per-layer results as a readable table string.

    Args:
        result: EvalResult from the evaluator.
        top_n: If set, only show the top N layers by layer_time_s.

    Returns:
        Formatted table string.
    """
    headers = [
        "Layer",
        "Op",
        "Engine",
        "MACs (M)",
        "DRAM (KB)",
        "Compute (ms)",
        "Memory (ms)",
        "Time (ms)",
        "Bottleneck",
    ]

    rows = []
    layer_results = result.layer_results

    if top_n is not None:
        # Sort by layer time descending, take top N
        sorted_layers = sorted(layer_results, key=lambda lr: lr.layer_time_s, reverse=True)
        layer_results = sorted_layers[:top_n]

    for lr in layer_results:
        if lr.is_fused:
            rows.append([
                _truncate(lr.name, 35),
                lr.op_type,
                lr.compute_type.upper(),
                f"{lr.macs / 1e6:.2f}",
                "-",
                "-",
                "-",
                "[fused]",
                "fused",
            ])
        else:
            rows.append([
                _truncate(lr.name, 35),
                lr.op_type,
                lr.compute_type.upper(),
                f"{lr.macs / 1e6:.2f}",
                f"{lr.dram_bytes / 1024:.1f}",
                f"{lr.compute_time_ms:.4f}",
                f"{lr.memory_time_ms:.4f}",
                f"{lr.layer_time_s * 1e3:.4f}",
                lr.bottleneck,
            ])

    return tabulate(rows, headers=headers, tablefmt="simple", numalign="right")


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string with ellipsis if needed."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(result: EvalResult, path: Union[str, Path]) -> None:
    """Export full evaluation result to a JSON file."""
    data = _result_to_dict(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _result_to_dict(result: EvalResult) -> dict:
    """Convert EvalResult to a JSON-serializable dict."""
    return {
        "chip": {
            "name": result.chip.name,
            "int8_tops": result.chip.int8_tops,
            "dram_bandwidth_gbps": result.chip.dram_bandwidth_gbps,
            "compute_efficiency": result.chip.compute_efficiency,
            "memory_efficiency": result.chip.memory_efficiency,
            "int8_tops_1d": result.chip.int8_tops_1d,
            "dram_bandwidth_gbps_1d": result.chip.dram_bandwidth_gbps_1d,
            "compute_efficiency_1d": result.chip.compute_efficiency_1d,
            "memory_efficiency_1d": result.chip.memory_efficiency_1d,
        },
        "model": result.model_name,
        "quantization": result.quantization,
        "input_resolution": {
            "height": result.input_resolution[0],
            "width": result.input_resolution[1],
        },
        "summary": {
            "total_macs": result.total_macs,
            "total_macs_G": round(result.total_macs / 1e9, 4),
            "total_ops": result.total_ops,
            "total_ops_G": round(result.total_ops / 1e9, 4),
            "total_dram_bytes": round(result.total_dram_bytes, 2),
            "total_dram_MB": round(result.total_dram_bytes / 1e6, 4),
            "total_inference_time_ms": round(result.total_time_ms, 4),
            "fps": round(result.fps, 2),
            "compute_bound_layers": result.compute_bound_layers,
            "memory_bound_layers": result.memory_bound_layers,
            "total_time_2d_ms": round(result.total_time_2d_s * 1e3, 4),
            "total_time_1d_ms": round(result.total_time_1d_s * 1e3, 4),
            "layers_2d": result.layers_2d,
            "layers_1d": result.layers_1d,
        },
        "layers": [_layer_to_dict(lr) for lr in result.layer_results],
    }


def _layer_to_dict(lr: LayerResult) -> dict:
    """Convert a LayerResult to a JSON-serializable dict."""
    return {
        "name": lr.name,
        "op_type": lr.op_type,
        "compute_type": lr.compute_type,
        "macs": lr.macs,
        "ops": lr.ops,
        "dram_bytes": round(lr.dram_bytes, 2),
        "compute_time_ms": round(lr.compute_time_ms, 6),
        "memory_time_ms": round(lr.memory_time_ms, 6),
        "layer_time_ms": round(lr.layer_time_ms, 6),
        "bottleneck": lr.bottleneck,
        "arithmetic_intensity": round(lr.arithmetic_intensity, 2),
        "is_fused": lr.is_fused,
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(result: EvalResult, path: Union[str, Path]) -> None:
    """Export per-layer results to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "layer_name",
        "op_type",
        "compute_type",
        "macs",
        "ops",
        "dram_bytes",
        "compute_time_ms",
        "memory_time_ms",
        "layer_time_ms",
        "bottleneck",
        "arithmetic_intensity",
        "is_fused",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lr in result.layer_results:
            writer.writerow({
                "layer_name": lr.name,
                "op_type": lr.op_type,
                "compute_type": lr.compute_type,
                "macs": lr.macs,
                "ops": lr.ops,
                "dram_bytes": round(lr.dram_bytes, 2),
                "compute_time_ms": round(lr.compute_time_ms, 6),
                "memory_time_ms": round(lr.memory_time_ms, 6),
                "layer_time_ms": round(lr.layer_time_ms, 6),
                "bottleneck": lr.bottleneck,
                "arithmetic_intensity": round(lr.arithmetic_intensity, 2),
                "is_fused": lr.is_fused,
            })

    # Also write a summary row at the end
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["# Summary"])
        writer.writerow(["# Model", result.model_name])
        writer.writerow(["# Chip", result.chip.name])
        writer.writerow(["# Quantization", result.quantization])
        writer.writerow([
            "# Resolution",
            f"{result.input_resolution[0]}x{result.input_resolution[1]}",
        ])
        writer.writerow(["# Total MACs (G)", f"{result.total_macs / 1e9:.4f}"])
        writer.writerow(["# Total DRAM (MB)", f"{result.total_dram_bytes / 1e6:.4f}"])
        writer.writerow(["# Inference time (ms)", f"{result.total_time_ms:.4f}"])
        writer.writerow(["# FPS", f"{result.fps:.2f}"])
