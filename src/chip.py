"""
ChipSpec: Hardware specification for an AI accelerator chip with dual compute paths.

The chip has two distinct compute units:
  - **2D Engine** (matrix multiply): handles Conv, Gemm, MatMul — high throughput
  - **1D Engine** (vector/special functions): handles activations, element-wise ops,
    pooling, softmax, etc. — lower throughput

Each engine has its own:
  - INT8 TOPS (peak throughput)
  - DRAM bandwidth (GB/s) to feed that engine
  - Compute efficiency (0-1)
  - Memory efficiency (0-1)

Quantization scaling rules (relative to INT8 TOPS, applied to both engines):
  - INT8:  1x TOPS,  1 byte per element
  - FP16:  0.5x TOPS (halved throughput), 2 bytes per element
  - INT4:  2x TOPS (doubled throughput), 0.5 bytes per element
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ChipSpec:
    """Hardware specification of an AI accelerator chip with 2D + 1D compute."""

    name: str

    # --- 2D Engine (matrix multiply: Conv, Gemm, MatMul) ---
    int8_tops: float              # Peak INT8 throughput in TOPS for 2D engine
    dram_bandwidth_gbps: float    # DRAM bandwidth in GB/s feeding the 2D engine
    compute_efficiency: float = 0.70   # Fraction of peak 2D compute achieved (0-1)
    memory_efficiency: float = 0.80    # Fraction of peak 2D bandwidth achieved (0-1)

    # --- 1D Engine (vector/special functions: activations, element-wise, pool) ---
    int8_tops_1d: float = 2.0            # Peak INT8 throughput in TOPS for 1D engine
    dram_bandwidth_gbps_1d: float = 120.0  # DRAM bandwidth in GB/s feeding 1D engine
    compute_efficiency_1d: float = 0.70    # Fraction of peak 1D compute achieved (0-1)
    memory_efficiency_1d: float = 0.80     # Fraction of peak 1D bandwidth achieved (0-1)

    # Quantization-to-TOPS multiplier relative to INT8 (shared by both engines)
    _QUANT_COMPUTE_SCALE: Dict[str, float] = field(
        default_factory=lambda: {
            "int8": 1.0,
            "fp16": 0.5,   # FP16 ops typically half the INT8 throughput
            "int4": 2.0,   # INT4 ops typically double the INT8 throughput
        },
        repr=False,
    )

    def _validate_quantization(self, quantization: str) -> float:
        """Return the quantization scale factor, raising ValueError if unsupported."""
        quant = quantization.lower()
        scale = self._QUANT_COMPUTE_SCALE.get(quant)
        if scale is None:
            raise ValueError(
                f"Unsupported quantization '{quantization}'. "
                f"Choose from: {list(self._QUANT_COMPUTE_SCALE.keys())}"
            )
        return scale

    # --- 2D engine effective metrics ---

    def effective_tops(self, quantization: str) -> float:
        """
        Return effective peak TOPS for the 2D engine at the given quantization.
        """
        scale = self._validate_quantization(quantization)
        return self.int8_tops * scale * self.compute_efficiency

    def effective_bandwidth_gbps(self) -> float:
        """Return effective DRAM bandwidth in GB/s for the 2D engine."""
        return self.dram_bandwidth_gbps * self.memory_efficiency

    # --- 1D engine effective metrics ---

    def effective_tops_1d(self, quantization: str) -> float:
        """
        Return effective peak TOPS for the 1D engine at the given quantization.
        """
        scale = self._validate_quantization(quantization)
        return self.int8_tops_1d * scale * self.compute_efficiency_1d

    def effective_bandwidth_gbps_1d(self) -> float:
        """Return effective DRAM bandwidth in GB/s for the 1D engine."""
        return self.dram_bandwidth_gbps_1d * self.memory_efficiency_1d

    @classmethod
    def from_dict(cls, d: dict) -> "ChipSpec":
        """Construct a ChipSpec from a dictionary."""
        return cls(
            name=d["name"],
            int8_tops=float(d["int8_tops"]),
            dram_bandwidth_gbps=float(d["dram_bandwidth_gbps"]),
            compute_efficiency=float(d.get("compute_efficiency", 0.70)),
            memory_efficiency=float(d.get("memory_efficiency", 0.80)),
            int8_tops_1d=float(d.get("int8_tops_1d", d.get("int8_tops", 10.0) * 0.1)),
            dram_bandwidth_gbps_1d=float(d.get("dram_bandwidth_gbps_1d",
                                                d.get("dram_bandwidth_gbps", 50.0) * 0.1)),
            compute_efficiency_1d=float(d.get("compute_efficiency_1d",
                                              d.get("compute_efficiency", 0.70))),
            memory_efficiency_1d=float(d.get("memory_efficiency_1d",
                                             d.get("memory_efficiency", 0.80))),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ChipSpec":
        """Load a single ChipSpec from a JSON file (first entry if it's a list)."""
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls.from_dict(data[0])
        return cls.from_dict(data)

    @classmethod
    def load_all(cls, path: str | Path) -> List["ChipSpec"]:
        """Load a list of ChipSpec from a JSON file that contains an array."""
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return [cls.from_dict(d) for d in data]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"Chip: {self.name}\n"
            f"  2D Engine — INT8 TOPS: {self.int8_tops:.1f}, "
            f"DRAM BW: {self.dram_bandwidth_gbps:.1f} GB/s, "
            f"η_compute: {self.compute_efficiency:.0%}, η_memory: {self.memory_efficiency:.0%}\n"
            f"  1D Engine — INT8 TOPS: {self.int8_tops_1d:.1f}, "
            f"DRAM BW: {self.dram_bandwidth_gbps_1d:.1f} GB/s, "
            f"η_compute: {self.compute_efficiency_1d:.0%}, η_memory: {self.memory_efficiency_1d:.0%}"
        )
