"""
Evaluator: Roofline-model-based performance estimator with dual compute paths.

The chip has two engines:
  - **2D Engine** (matrix multiply): Conv, Gemm, MatMul — high TOPS, high bandwidth
  - **1D Engine** (vector/special): activations, element-wise, pool — lower TOPS/bandwidth

For each layer, the execution time is bounded by:
  layer_time = max(compute_time, memory_time)

using the engine (2D or 1D) that matches the layer's compute_type.

Total inference time is the sum of all layer times (sequential execution,
batch=1).  FPS = 1 / total_time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .chip import ChipSpec
from .layer import LayerProfile


@dataclass
class LayerResult:
    """Per-layer evaluation result."""

    name: str
    op_type: str
    macs: int
    ops: int
    dram_bytes: float            # Total DRAM read + write (bytes)
    compute_time_s: float        # Seconds (compute-limited)
    memory_time_s: float         # Seconds (memory-limited)
    layer_time_s: float          # Actual time = max(compute, memory)
    bottleneck: str              # "compute" or "memory"
    arithmetic_intensity: float  # OPS / byte
    is_fused: bool
    compute_type: str = "1d"     # "2d" (matrix engine) or "1d" (vector engine)

    @property
    def layer_time_ms(self) -> float:
        return self.layer_time_s * 1e3

    @property
    def compute_time_ms(self) -> float:
        return self.compute_time_s * 1e3

    @property
    def memory_time_ms(self) -> float:
        return self.memory_time_s * 1e3


@dataclass
class EvalResult:
    """Complete evaluation result for a model on a chip."""

    chip: ChipSpec
    model_name: str
    quantization: str
    input_resolution: Tuple[int, int]
    layer_results: List[LayerResult] = field(default_factory=list)

    # Summary (populated after evaluation)
    total_macs: int = 0
    total_ops: int = 0
    total_dram_bytes: float = 0.0
    total_time_s: float = 0.0
    fps: float = 0.0
    total_compute_time_s: float = 0.0
    total_memory_time_s: float = 0.0
    compute_bound_layers: int = 0
    memory_bound_layers: int = 0

    # Per-engine breakdown
    total_time_2d_s: float = 0.0   # Time spent on 2D (matrix) engine
    total_time_1d_s: float = 0.0   # Time spent on 1D (vector) engine
    layers_2d: int = 0
    layers_1d: int = 0

    @property
    def total_time_ms(self) -> float:
        return self.total_time_s * 1e3

    def summary_str(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 70,
            f"  Model       : {self.model_name}",
            f"  Quantization: {self.quantization.upper()}",
            f"  Resolution  : {self.input_resolution[0]}x{self.input_resolution[1]}",
            f"  Chip        : {self.chip.name}",
            f"    2D Engine — INT8 TOPS: {self.chip.int8_tops:.1f}, "
            f"DRAM BW: {self.chip.dram_bandwidth_gbps:.1f} GB/s",
            f"    1D Engine — INT8 TOPS: {self.chip.int8_tops_1d:.1f}, "
            f"DRAM BW: {self.chip.dram_bandwidth_gbps_1d:.1f} GB/s",
            "-" * 70,
            f"  Total MACs          : {self.total_macs / 1e9:>10.3f} G",
            f"  Total OPS           : {self.total_ops / 1e9:>10.3f} G",
            f"  Total DRAM traffic  : {self.total_dram_bytes / 1e6:>10.3f} MB",
            f"  Compute time (sum)  : {self.total_compute_time_s * 1e3:>10.3f} ms",
            f"  Memory time (sum)   : {self.total_memory_time_s * 1e3:>10.3f} ms",
            f"  Total inference time: {self.total_time_ms:>10.3f} ms",
            f"    2D engine time    : {self.total_time_2d_s * 1e3:>10.3f} ms  ({self.layers_2d} layers)",
            f"    1D engine time    : {self.total_time_1d_s * 1e3:>10.3f} ms  ({self.layers_1d} layers)",
            "-" * 70,
            f"  >>> FPS (batch=1)   : {self.fps:>10.2f} <<<",
            "-" * 70,
            f"  Compute-bound layers: {self.compute_bound_layers}",
            f"  Memory-bound layers : {self.memory_bound_layers}",
            "=" * 70,
        ]
        return "\n".join(lines)


def evaluate(
    chip: ChipSpec,
    layers: List[LayerProfile],
    quantization: str,
    input_resolution: Tuple[int, int],
    model_name: str = "unknown",
) -> EvalResult:
    """
    Run the roofline evaluation.

    Args:
        chip: Hardware specification (with 2D + 1D engine specs).
        layers: List of layer profiles from the ONNX parser.
        quantization: 'int8', 'fp16', or 'int4'.
        input_resolution: (H, W) of the input image.
        model_name: Name of the model (for display purposes).

    Returns:
        EvalResult with per-layer breakdown and overall FPS.
    """
    # --- 2D engine (matrix multiply: Conv, Gemm, MatMul) ---
    eff_tops_2d = chip.effective_tops(quantization)              # TOPS
    eff_bw_2d = chip.effective_bandwidth_gbps()                  # GB/s
    eff_ops_per_sec_2d = eff_tops_2d * 1e12
    eff_bytes_per_sec_2d = eff_bw_2d * 1e9

    # --- 1D engine (vector/special: activations, element-wise, pool, etc.) ---
    eff_tops_1d = chip.effective_tops_1d(quantization)           # TOPS
    eff_bw_1d = chip.effective_bandwidth_gbps_1d()               # GB/s
    eff_ops_per_sec_1d = eff_tops_1d * 1e12
    eff_bytes_per_sec_1d = eff_bw_1d * 1e9

    result = EvalResult(
        chip=chip,
        model_name=model_name,
        quantization=quantization,
        input_resolution=input_resolution,
    )

    for layer in layers:
        ops = layer.ops
        dram_bytes = layer.total_dram_bytes(quantization)
        ctype = layer.compute_type  # "2d" or "1d"

        # Select the appropriate engine's throughput and bandwidth
        if ctype == "2d":
            eff_ops_per_sec = eff_ops_per_sec_2d
            eff_bytes_per_sec = eff_bytes_per_sec_2d
        else:
            eff_ops_per_sec = eff_ops_per_sec_1d
            eff_bytes_per_sec = eff_bytes_per_sec_1d

        if layer.is_fused:
            # Fused layers contribute zero time
            lr = LayerResult(
                name=layer.name,
                op_type=layer.op_type,
                macs=layer.macs,
                ops=ops,
                dram_bytes=dram_bytes,
                compute_time_s=0.0,
                memory_time_s=0.0,
                layer_time_s=0.0,
                bottleneck="fused",
                arithmetic_intensity=layer.arithmetic_intensity(quantization),
                is_fused=True,
                compute_type=ctype,
            )
        else:
            # Compute time (seconds)
            if ops > 0:
                compute_time = ops / eff_ops_per_sec
            else:
                compute_time = 0.0

            # Memory time (seconds)
            if dram_bytes > 0:
                memory_time = dram_bytes / eff_bytes_per_sec
            else:
                memory_time = 0.0

            # Roofline: actual time = max(compute, memory)
            layer_time = max(compute_time, memory_time)

            if compute_time >= memory_time:
                bottleneck = "compute"
            else:
                bottleneck = "memory"

            lr = LayerResult(
                name=layer.name,
                op_type=layer.op_type,
                macs=layer.macs,
                ops=ops,
                dram_bytes=dram_bytes,
                compute_time_s=compute_time,
                memory_time_s=memory_time,
                layer_time_s=layer_time,
                bottleneck=bottleneck,
                arithmetic_intensity=layer.arithmetic_intensity(quantization),
                is_fused=False,
                compute_type=ctype,
            )

        result.layer_results.append(lr)

    # Aggregate
    result.total_macs = sum(lr.macs for lr in result.layer_results)
    result.total_ops = sum(lr.ops for lr in result.layer_results)
    result.total_dram_bytes = sum(lr.dram_bytes for lr in result.layer_results)
    result.total_compute_time_s = sum(lr.compute_time_s for lr in result.layer_results)
    result.total_memory_time_s = sum(lr.memory_time_s for lr in result.layer_results)
    result.total_time_s = sum(lr.layer_time_s for lr in result.layer_results)
    result.fps = 1.0 / result.total_time_s if result.total_time_s > 0 else float("inf")
    result.compute_bound_layers = sum(
        1 for lr in result.layer_results if lr.bottleneck == "compute"
    )
    result.memory_bound_layers = sum(
        1 for lr in result.layer_results if lr.bottleneck == "memory"
    )

    # Per-engine breakdown
    result.total_time_2d_s = sum(
        lr.layer_time_s for lr in result.layer_results if lr.compute_type == "2d"
    )
    result.total_time_1d_s = sum(
        lr.layer_time_s for lr in result.layer_results if lr.compute_type == "1d"
    )
    result.layers_2d = sum(
        1 for lr in result.layer_results
        if lr.compute_type == "2d" and not lr.is_fused
    )
    result.layers_1d = sum(
        1 for lr in result.layer_results
        if lr.compute_type == "1d" and not lr.is_fused
    )

    return result
