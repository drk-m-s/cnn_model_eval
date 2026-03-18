"""
Evaluator: Roofline-model-based performance estimator.

For each layer, the execution time is bounded by:
  layer_time = max(compute_time, memory_time)

where:
  compute_time = layer.ops / effective_tops
  memory_time  = layer.dram_bytes / effective_bandwidth

The total inference time is the sum of all layer times (sequential execution,
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
            f"    INT8 TOPS : {self.chip.int8_tops:.1f}",
            f"    DRAM BW   : {self.chip.dram_bandwidth_gbps:.1f} GB/s",
            f"    Compute η : {self.chip.compute_efficiency:.0%}",
            f"    Memory η  : {self.chip.memory_efficiency:.0%}",
            "-" * 70,
            f"  Total MACs          : {self.total_macs / 1e9:>10.3f} G",
            f"  Total OPS           : {self.total_ops / 1e9:>10.3f} G",
            f"  Total DRAM traffic  : {self.total_dram_bytes / 1e6:>10.3f} MB",
            f"  Compute time (sum)  : {self.total_compute_time_s * 1e3:>10.3f} ms",
            f"  Memory time (sum)   : {self.total_memory_time_s * 1e3:>10.3f} ms",
            f"  Total inference time: {self.total_time_ms:>10.3f} ms",
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
        chip: Hardware specification.
        layers: List of layer profiles from the ONNX parser.
        quantization: 'int8', 'fp16', or 'int4'.
        input_resolution: (H, W) of the input image.
        model_name: Name of the model (for display purposes).

    Returns:
        EvalResult with per-layer breakdown and overall FPS.
    """
    effective_tops = chip.effective_tops(quantization)           # TOPS after efficiency
    effective_bw = chip.effective_bandwidth_gbps()                # GB/s after efficiency

    # Convert to base units
    effective_ops_per_sec = effective_tops * 1e12       # operations per second
    effective_bytes_per_sec = effective_bw * 1e9        # bytes per second

    result = EvalResult(
        chip=chip,
        model_name=model_name,
        quantization=quantization,
        input_resolution=input_resolution,
    )

    for layer in layers:
        ops = layer.ops
        dram_bytes = layer.total_dram_bytes(quantization)

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
            )
        else:
            # Compute time (seconds)
            if ops > 0:
                compute_time = ops / effective_ops_per_sec
            else:
                compute_time = 0.0

            # Memory time (seconds)
            if dram_bytes > 0:
                memory_time = dram_bytes / effective_bytes_per_sec
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

    return result
