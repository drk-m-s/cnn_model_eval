"""
ChipSpec: Hardware specification for an AI accelerator chip.

The chip is characterized by:
  - INT8 TOPS (peak throughput for INT8 operations)
  - DRAM bandwidth (GB/s)
  - Compute efficiency (0-1): fraction of peak TOPS actually achievable
  - Memory efficiency (0-1): fraction of peak bandwidth actually achievable

Quantization scaling rules (relative to INT8 TOPS):
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
    """Hardware specification of an AI accelerator chip."""

    name: str
    int8_tops: float          # Peak INT8 throughput in TOPS (tera-operations per second)
    dram_bandwidth_gbps: float  # DRAM bandwidth in GB/s
    compute_efficiency: float = 0.70  # Achievable fraction of peak compute (0-1)
    memory_efficiency: float = 0.80   # Achievable fraction of peak bandwidth (0-1)

    # Quantization-to-TOPS multiplier relative to INT8
    _QUANT_COMPUTE_SCALE: Dict[str, float] = field(
        default_factory=lambda: {
            "int8": 1.0,
            "fp16": 0.5,   # FP16 ops typically half the INT8 throughput
            "int4": 2.0,   # INT4 ops typically double the INT8 throughput
        },
        repr=False,
    )

    def effective_tops(self, quantization: str) -> float:
        """
        Return effective peak TOPS for the given quantization, considering the
        chip's compute efficiency.

        Args:
            quantization: One of 'int8', 'fp16', 'int4'.

        Returns:
            Effective TOPS after applying quantization scaling and compute efficiency.
        """
        quant = quantization.lower()
        scale = self._QUANT_COMPUTE_SCALE.get(quant)
        if scale is None:
            raise ValueError(
                f"Unsupported quantization '{quantization}'. "
                f"Choose from: {list(self._QUANT_COMPUTE_SCALE.keys())}"
            )
        return self.int8_tops * scale * self.compute_efficiency

    def effective_bandwidth_gbps(self) -> float:
        """Return effective DRAM bandwidth in GB/s after applying memory efficiency."""
        return self.dram_bandwidth_gbps * self.memory_efficiency

    @classmethod
    def from_dict(cls, d: dict) -> "ChipSpec":
        """Construct a ChipSpec from a dictionary."""
        return cls(
            name=d["name"],
            int8_tops=float(d["int8_tops"]),
            dram_bandwidth_gbps=float(d["dram_bandwidth_gbps"]),
            compute_efficiency=float(d.get("compute_efficiency", 0.70)),
            memory_efficiency=float(d.get("memory_efficiency", 0.80)),
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
            f"  INT8 Peak TOPS    : {self.int8_tops:.1f}\n"
            f"  DRAM Bandwidth    : {self.dram_bandwidth_gbps:.1f} GB/s\n"
            f"  Compute Efficiency: {self.compute_efficiency:.0%}\n"
            f"  Memory Efficiency : {self.memory_efficiency:.0%}"
        )
