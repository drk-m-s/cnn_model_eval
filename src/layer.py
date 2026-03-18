"""
LayerProfile: Per-layer description extracted from an ONNX model.

Stores the operation type, shapes, parameter counts, and computed
MAC / OPS / DRAM-bytes figures that the evaluator needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Bytes per element for each quantization type
BYTES_PER_ELEMENT: Dict[str, float] = {
    "int8": 1.0,
    "fp16": 2.0,
    "int4": 0.5,
}


def _product(shape: List[int]) -> int:
    """Return the product of a shape list, or 0 if empty."""
    result = 1
    for s in shape:
        result *= s
    return result if shape else 0


@dataclass
class LayerProfile:
    """Profile of a single layer / node in the computational graph."""

    name: str                          # Node name (or generated ID)
    op_type: str                       # ONNX op type (Conv, Gemm, Relu, etc.)

    # Shapes  (batch dimension included, e.g. [1, 64, 112, 112])
    input_shapes: List[List[int]] = field(default_factory=list)
    output_shapes: List[List[int]] = field(default_factory=list)

    # Conv / Pool attributes (None for non-conv layers)
    kernel_size: Optional[Tuple[int, int]] = None
    strides: Optional[Tuple[int, int]] = None
    pads: Optional[Tuple[int, ...]] = None
    groups: int = 1

    # Parameter counts (number of elements, not bytes)
    weight_params: int = 0
    bias_params: int = 0

    # Computed workload
    macs: int = 0           # Multiply-accumulate operations
    is_fused: bool = False  # If True, this layer is fused into a preceding layer (0 cost)

    @property
    def ops(self) -> int:
        """Total arithmetic operations (each MAC = 2 OPS: one multiply + one add)."""
        return self.macs * 2

    @property
    def primary_input_shape(self) -> List[int]:
        """First input tensor's shape (activation input)."""
        return self.input_shapes[0] if self.input_shapes else []

    @property
    def primary_output_shape(self) -> List[int]:
        """First output tensor's shape."""
        return self.output_shapes[0] if self.output_shapes else []

    def input_activation_elements(self) -> int:
        """Number of elements in the primary input activation tensor."""
        return _product(self.primary_input_shape)

    def output_activation_elements(self) -> int:
        """Number of elements in the primary output activation tensor."""
        return _product(self.primary_output_shape)

    def dram_read_bytes(self, quantization: str) -> float:
        """
        Estimated DRAM bytes READ for this layer.

        Includes:
          - Input activation (read from DRAM)
          - Weights (read from DRAM)
          - Bias (read from DRAM)

        For batch=1, we assume no on-chip caching of activations across layers
        (conservative / worst-case DRAM traffic estimate).
        """
        if self.is_fused:
            return 0.0
        bpe = BYTES_PER_ELEMENT.get(quantization.lower(), 1.0)
        activation_bytes = self.input_activation_elements() * bpe
        weight_bytes = self.weight_params * bpe
        bias_bytes = self.bias_params * bpe  # bias often stored at higher precision, but we simplify
        return activation_bytes + weight_bytes + bias_bytes

    def dram_write_bytes(self, quantization: str) -> float:
        """
        Estimated DRAM bytes WRITTEN for this layer (output activation).
        """
        if self.is_fused:
            return 0.0
        bpe = BYTES_PER_ELEMENT.get(quantization.lower(), 1.0)
        return self.output_activation_elements() * bpe

    def total_dram_bytes(self, quantization: str) -> float:
        """Total DRAM traffic (read + write) in bytes."""
        return self.dram_read_bytes(quantization) + self.dram_write_bytes(quantization)

    def arithmetic_intensity(self, quantization: str) -> float:
        """
        Arithmetic intensity = OPS / DRAM bytes.

        A higher value means the layer is more compute-bound.
        """
        dram = self.total_dram_bytes(quantization)
        if dram == 0:
            return float("inf")
        return self.ops / dram

    def __repr__(self) -> str:
        fused_tag = " [FUSED]" if self.is_fused else ""
        return (
            f"LayerProfile(name={self.name!r}, op={self.op_type}, "
            f"MACs={self.macs:,}, weight_params={self.weight_params:,}{fused_tag})"
        )
