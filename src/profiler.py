"""
Profiler utilities: compute MACs and DRAM byte estimates for common layer types.

These functions are used by the ONNX parser to fill in LayerProfile fields.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .layer import BYTES_PER_ELEMENT


# ---------------------------------------------------------------------------
# MAC computation helpers
# ---------------------------------------------------------------------------

def compute_conv_macs(
    input_shape: List[int],      # [N, C_in, H_in, W_in]
    weight_shape: List[int],     # [C_out, C_in/groups, K_h, K_w]
    output_shape: List[int],     # [N, C_out, H_out, W_out]
    groups: int = 1,
    has_bias: bool = False,
) -> int:
    """
    Compute MACs for a 2D convolution layer.

    MACs = H_out * W_out * C_out * K_h * K_w * (C_in / groups)
    If bias: additionally C_out * H_out * W_out (negligible, but included).
    """
    if len(output_shape) < 4 or len(weight_shape) < 4:
        return 0

    _, c_out, h_out, w_out = output_shape[:4]
    _, c_in_per_group, k_h, k_w = weight_shape[:4]

    macs = h_out * w_out * c_out * k_h * k_w * c_in_per_group
    if has_bias:
        macs += c_out * h_out * w_out  # bias addition
    return int(macs)


def compute_conv_transpose_macs(
    input_shape: List[int],      # [N, C_in, H_in, W_in]
    weight_shape: List[int],     # [C_in, C_out/groups, K_h, K_w]
    output_shape: List[int],     # [N, C_out, H_out, W_out]
    groups: int = 1,
    has_bias: bool = False,
) -> int:
    """
    Compute MACs for a 2D transposed convolution layer.

    ConvTranspose weight shape is [C_in, C_out/groups, K_h, K_w].
    MACs = H_out * W_out * C_out * K_h * K_w * (C_in / groups)
    """
    if len(output_shape) < 4 or len(weight_shape) < 4:
        return 0

    _, c_out, h_out, w_out = output_shape[:4]
    c_in = weight_shape[0]
    _, _, k_h, k_w = weight_shape[:4]

    c_in_per_group = c_in // groups if groups > 0 else c_in
    macs = h_out * w_out * c_out * k_h * k_w * c_in_per_group
    if has_bias:
        macs += c_out * h_out * w_out
    return int(macs)


def compute_gemm_macs(M: int, K: int, N: int, has_bias: bool = False) -> int:
    """
    Compute MACs for a fully-connected / Gemm layer.

    MACs = M * K * N  (matrix multiply)
    """
    macs = M * K * N
    if has_bias:
        macs += M * N
    return int(macs)


def compute_matmul_macs(a_shape: List[int], b_shape: List[int]) -> int:
    """
    Compute MACs for a MatMul node.

    For 2D: [M, K] x [K, N] → M * K * N
    For batched: batch dims are multiplied element-wise.
    """
    if len(a_shape) < 2 or len(b_shape) < 2:
        return 0

    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]

    batch = 1
    for dim in a_shape[:-2]:
        batch *= dim

    return int(batch * M * K * N)


def compute_pool_macs(
    output_shape: List[int],
    kernel_size: Tuple[int, int],
) -> int:
    """
    Compute approximate MACs for a pooling layer.

    Pool layers perform comparisons or additions over the kernel window.
    MACs ≈ H_out * W_out * C * K_h * K_w  (much less than Conv, but nonzero).
    """
    if len(output_shape) < 4:
        return 0
    _, c, h_out, w_out = output_shape[:4]
    k_h, k_w = kernel_size
    return int(c * h_out * w_out * k_h * k_w)


def compute_element_wise_macs(output_shape: List[int]) -> int:
    """
    MACs for an element-wise operation (Relu, Add, Mul, Sigmoid, etc.).

    Approximately 1 operation per element (we count 1 MAC per element for
    simplicity, even though some ops like Sigmoid are more expensive).
    """
    result = 1
    for d in output_shape:
        result *= d
    return int(result)


def compute_batchnorm_macs(output_shape: List[int]) -> int:
    """
    MACs for BatchNormalization (when NOT fused with Conv).

    ~4 ops per element: subtract mean, divide by std, scale, shift.
    We count ~2 MACs per element.
    """
    elements = 1
    for d in output_shape:
        elements *= d
    return int(elements * 2)


# ---------------------------------------------------------------------------
# DRAM byte helpers
# ---------------------------------------------------------------------------

def bytes_per_element(quantization: str) -> float:
    """Return bytes per element for a quantization type."""
    return BYTES_PER_ELEMENT.get(quantization.lower(), 1.0)


def compute_dram_bytes(
    input_elements: int,
    output_elements: int,
    weight_elements: int,
    bias_elements: int,
    quantization: str,
) -> float:
    """
    Compute total DRAM traffic in bytes.

    Total = (input_act + weights + bias) [read] + output_act [write]
    """
    bpe = bytes_per_element(quantization)
    read_bytes = (input_elements + weight_elements + bias_elements) * bpe
    write_bytes = output_elements * bpe
    return read_bytes + write_bytes
