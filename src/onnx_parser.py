"""
ONNX Parser: Load an ONNX model and extract per-layer profiles.

Handles the key CNN operations:
  - Conv, ConvTranspose
  - Gemm, MatMul
  - MaxPool, AveragePool, GlobalAveragePool
  - BatchNormalization (marked as fused when following Conv)
  - Relu, LeakyRelu, Sigmoid, Clip (ReLU6), Mul (SiLU pattern), Add, Concat
  - Resize, Upsample
  - Reshape, Flatten, Transpose, Squeeze, Unsqueeze (shape-only, zero compute)
  - Softmax
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference

from .layer import LayerProfile
from .profiler import (
    compute_batchnorm_macs,
    compute_conv_macs,
    compute_conv_transpose_macs,
    compute_element_wise_macs,
    compute_gemm_macs,
    compute_matmul_macs,
    compute_pool_macs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attribute(node: Any, attr_name: str, default: Any = None) -> Any:
    """Extract a named attribute from an ONNX node."""
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
            elif attr.type == onnx.AttributeProto.TENSOR:
                return numpy_helper.to_array(attr.t)
            else:
                return default
    return default


def _shape_from_type_proto(type_proto) -> List[int]:
    """Extract shape as a list of ints from an ONNX TypeProto."""
    shape = []
    if type_proto.HasField("tensor_type") and type_proto.tensor_type.HasField("shape"):
        for dim in type_proto.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                # Dynamic dimension (e.g., batch) — assume 1
                shape.append(1)
    return shape


def _resolve_input_resolution(
    model: onnx.ModelProto,
    input_resolution: Optional[Tuple[int, int]],
) -> onnx.ModelProto:
    """
    If input_resolution is provided, override the model's spatial input dims.
    This allows evaluating the model at arbitrary resolutions.
    """
    if input_resolution is None:
        return model

    graph = model.graph
    for inp in graph.input:
        shape = _shape_from_type_proto(inp.type)
        if len(shape) == 4:
            # Shape is [N, C, H, W] — override H and W
            h, w = input_resolution
            dim = inp.type.tensor_type.shape.dim
            dim[0].dim_value = 1   # batch = 1
            dim[2].dim_value = h
            dim[3].dim_value = w
            logger.info(
                "Overriding input '%s' to shape [1, %d, %d, %d]",
                inp.name, shape[1], h, w,
            )
    return model


# ---------------------------------------------------------------------------
# ONNX shape/initializer maps
# ---------------------------------------------------------------------------

class _GraphInfo:
    """Pre-computed lookup tables for an ONNX graph."""

    def __init__(self, graph: Any):
        self.graph = graph
        # tensor name → shape (list of ints)
        self.shape_map: Dict[str, List[int]] = {}
        # tensor name → numpy array (for weights / initializers)
        self.initializer_map: Dict[str, np.ndarray] = {}
        # tensor name → number of elements
        self.initializer_sizes: Dict[str, int] = {}
        # set of initializer names (weights, not activations)
        self.initializer_names: set = set()
        # node output name → node (for fusion detection)
        self.producer_map: Dict[str, Any] = {}

        self._build_initializer_map()
        self._build_shape_map()
        self._build_producer_map()

    def _build_initializer_map(self):
        for init in self.graph.initializer:
            arr = numpy_helper.to_array(init)
            self.initializer_map[init.name] = arr
            self.initializer_sizes[init.name] = int(arr.size)
            self.initializer_names.add(init.name)
            self.shape_map[init.name] = list(arr.shape)

    def _build_shape_map(self):
        # From graph inputs
        for inp in self.graph.input:
            shape = _shape_from_type_proto(inp.type)
            if shape:
                self.shape_map[inp.name] = shape

        # From value_info (intermediate tensors after shape inference)
        for vi in self.graph.value_info:
            shape = _shape_from_type_proto(vi.type)
            if shape:
                self.shape_map[vi.name] = shape

        # From graph outputs
        for out in self.graph.output:
            shape = _shape_from_type_proto(out.type)
            if shape:
                self.shape_map[out.name] = shape

    def _build_producer_map(self):
        for node in self.graph.node:
            for output_name in node.output:
                self.producer_map[output_name] = node

    def get_shape(self, tensor_name: str) -> List[int]:
        """Return shape for a tensor name, or empty list if unknown."""
        return self.shape_map.get(tensor_name, [])

    def get_input_shapes(self, node) -> List[List[int]]:
        """Return shapes for all inputs of a node."""
        return [self.get_shape(inp) for inp in node.input]

    def get_output_shapes(self, node) -> List[List[int]]:
        """Return shapes for all outputs of a node."""
        return [self.get_shape(out) for out in node.output]

    def get_weight_shape(self, node, idx: int = 1) -> List[int]:
        """Return shape of the weight input (typically index 1 for Conv/Gemm)."""
        if len(node.input) > idx:
            return self.get_shape(node.input[idx])
        return []

    def get_weight_elements(self, node, idx: int = 1) -> int:
        """Return number of weight elements (parameter count) for input at idx."""
        if len(node.input) > idx and node.input[idx] in self.initializer_sizes:
            return self.initializer_sizes[node.input[idx]]
        # Fallback: compute from shape
        shape = self.get_weight_shape(node, idx)
        result = 1
        for d in shape:
            result *= d
        return result if shape else 0

    def get_bias_elements(self, node, idx: int = 2) -> int:
        """Return number of bias elements for input at idx."""
        if len(node.input) > idx and node.input[idx] != "":
            return self.initializer_sizes.get(node.input[idx], 0)
        return 0

    def is_initializer(self, tensor_name: str) -> bool:
        """Check if a tensor name is an initializer (weight/constant)."""
        return tensor_name in self.initializer_names


# ---------------------------------------------------------------------------
# Node handlers
# ---------------------------------------------------------------------------


def _compute_activation_elements(node, info: _GraphInfo) -> int:
    """Sum elements of all non-initializer inputs (i.e., activation tensors read from DRAM)."""
    total = 0
    for inp_name in node.input:
        if inp_name and not info.is_initializer(inp_name):
            shape = info.get_shape(inp_name)
            if shape:
                elems = 1
                for d in shape:
                    elems *= d
                total += elems
    return total


def _handle_conv(node, info: _GraphInfo) -> LayerProfile:
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)
    weight_shape = info.get_weight_shape(node, idx=1)

    kernel = _get_attribute(node, "kernel_shape", [1, 1])
    strides = _get_attribute(node, "strides", [1, 1])
    pads = _get_attribute(node, "pads", [0, 0, 0, 0])
    groups = _get_attribute(node, "group", 1)

    weight_params = info.get_weight_elements(node, idx=1)
    bias_params = info.get_bias_elements(node, idx=2)
    has_bias = bias_params > 0

    out_shape = output_shapes[0] if output_shapes else []
    if node.op_type == "ConvTranspose":
        macs = compute_conv_transpose_macs(
            input_shape=input_shapes[0] if input_shapes else [],
            weight_shape=weight_shape,
            output_shape=out_shape,
            groups=groups,
            has_bias=has_bias,
        )
    else:
        macs = compute_conv_macs(
            input_shape=input_shapes[0] if input_shapes else [],
            weight_shape=weight_shape,
            output_shape=out_shape,
            groups=groups,
            has_bias=has_bias,
        )

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        kernel_size=tuple(kernel) if kernel else None,
        strides=tuple(strides) if strides else None,
        pads=tuple(pads) if pads else None,
        groups=groups,
        weight_params=weight_params,
        bias_params=bias_params,
        macs=macs,
    )


def _handle_gemm(node, info: _GraphInfo) -> LayerProfile:
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)

    weight_params = info.get_weight_elements(node, idx=1)
    bias_params = info.get_bias_elements(node, idx=2)

    # Determine M, K, N
    a_shape = input_shapes[0] if input_shapes else []
    b_shape = input_shapes[1] if len(input_shapes) > 1 else []

    transA = _get_attribute(node, "transA", 0)
    transB = _get_attribute(node, "transB", 0)

    if len(a_shape) >= 2:
        M = a_shape[-2] if not transA else a_shape[-1]
        K = a_shape[-1] if not transA else a_shape[-2]
    else:
        M, K = 1, 1

    if len(b_shape) >= 2:
        N = b_shape[-1] if not transB else b_shape[-2]
    else:
        N = 1

    macs = compute_gemm_macs(M, K, N, has_bias=bias_params > 0)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        weight_params=weight_params,
        bias_params=bias_params,
        macs=macs,
    )


def _handle_matmul(node, info: _GraphInfo) -> LayerProfile:
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)

    a_shape = input_shapes[0] if input_shapes else []
    b_shape = input_shapes[1] if len(input_shapes) > 1 else []

    # Weights: whichever input is an initializer
    weight_params = 0
    for idx in [0, 1]:
        if len(node.input) > idx and info.is_initializer(node.input[idx]):
            weight_params += info.get_weight_elements(node, idx)

    macs = compute_matmul_macs(a_shape, b_shape)
    act_elements = _compute_activation_elements(node, info)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        weight_params=weight_params,
        macs=macs,
        activation_read_elements=act_elements,
    )


def _handle_pool(node, info: _GraphInfo) -> LayerProfile:
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)

    out_shape = output_shapes[0] if output_shapes else []

    if node.op_type == "GlobalAveragePool":
        # Kernel covers entire spatial dims
        in_shape = input_shapes[0] if input_shapes else []
        if len(in_shape) >= 4:
            kernel = (in_shape[2], in_shape[3])
        else:
            kernel = (1, 1)
    else:
        kernel = tuple(_get_attribute(node, "kernel_shape", [1, 1]))

    strides = _get_attribute(node, "strides", [1, 1])
    macs = compute_pool_macs(out_shape, kernel)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        kernel_size=kernel,
        strides=tuple(strides) if strides else None,
        macs=macs,
        ops_override=macs,  # Pool: 1 op per window element (compare/add), not 2
    )


def _handle_batchnorm(node, info: _GraphInfo) -> LayerProfile:
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)

    out_shape = output_shapes[0] if output_shapes else []

    # BN parameters: scale, bias, mean, var (inputs 1-4)
    weight_params = 0
    bias_params = 0
    for idx in range(1, min(5, len(node.input))):
        elems = info.get_weight_elements(node, idx)
        if idx == 1:
            weight_params += elems  # scale (gamma)
        elif idx == 2:
            bias_params += elems   # bias (beta)
        else:
            weight_params += elems  # mean, var

    # Fuse BN with preceding Conv (check actual data flow, not just node order)
    producer = info.producer_map.get(node.input[0]) if node.input else None
    is_fused = producer is not None and producer.op_type in ("Conv", "ConvTranspose")

    macs = 0 if is_fused else compute_batchnorm_macs(out_shape)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        weight_params=weight_params,
        bias_params=bias_params,
        macs=macs,
        is_fused=is_fused,
    )


# OPS per element for activation functions (used instead of blanket macs*2)
_ACTIVATION_OPS_PER_ELEMENT = {
    "Relu": 1,          # comparison
    "Clip": 1,          # comparison (ReLU6)
    "LeakyRelu": 2,     # compare + conditional multiply
    "Sigmoid": 4,       # negate, exp, add, divide
    "Tanh": 5,          # ~2*sigmoid(2x) - 1
    "Softmax": 5,       # max, subtract, exp, sum, divide (per element)
    "Elu": 4,           # compare, subtract, exp, multiply
    "HardSigmoid": 3,   # multiply, add, clip
    "HardSwish": 4,     # multiply, add, clip, multiply
}


def _handle_activation(node, info: _GraphInfo) -> LayerProfile:
    """Handle Relu, LeakyRelu, Sigmoid, Clip, Softmax, etc."""
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)
    out_shape = output_shapes[0] if output_shapes else []
    elements = compute_element_wise_macs(out_shape)
    ops_per_elem = _ACTIVATION_OPS_PER_ELEMENT.get(node.op_type, 1)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        macs=elements,
        ops_override=elements * ops_per_elem,
    )


def _handle_elementwise(node, info: _GraphInfo) -> LayerProfile:
    """Handle Add, Mul, Sub, Div — element-wise binary ops (1 op per element)."""
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)
    out_shape = output_shapes[0] if output_shapes else []
    elements = compute_element_wise_macs(out_shape)

    # Check if any input is an initializer (learned parameter)
    weight_params = 0
    for idx, inp_name in enumerate(node.input):
        if info.is_initializer(inp_name):
            weight_params += info.initializer_sizes.get(inp_name, 0)

    act_elements = _compute_activation_elements(node, info)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        weight_params=weight_params,
        macs=elements,
        ops_override=elements,  # 1 op per element for Add/Sub/Mul/Div
        activation_read_elements=act_elements,
    )


def _handle_concat(node, info: _GraphInfo) -> LayerProfile:
    """Handle Concat — pure data movement, zero arithmetic."""
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)
    act_elements = _compute_activation_elements(node, info)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        macs=0,
        ops_override=0,
        activation_read_elements=act_elements,
    )


def _handle_resize(node, info: _GraphInfo) -> LayerProfile:
    """Handle Resize, Upsample — mode-dependent compute."""
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)
    out_shape = output_shapes[0] if output_shapes else []
    act_elements = _compute_activation_elements(node, info)

    out_elements = 1
    for d in out_shape:
        out_elements *= d
    out_elements = out_elements if out_shape else 0

    mode = _get_attribute(node, "mode", "nearest")
    if mode == "nearest":
        ops = 0  # index lookup only
    elif mode == "linear":
        ops = out_elements * 7  # bilinear: 4 mul + 3 add per element
    elif mode == "cubic":
        ops = out_elements * 40  # bicubic: ~36 mul + adds per element
    else:
        ops = out_elements  # conservative fallback

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        macs=0,
        ops_override=ops,
        activation_read_elements=act_elements,
    )


def _handle_reshape_like(node, info: _GraphInfo) -> LayerProfile:
    """Handle Reshape, Flatten, Transpose, Squeeze, Unsqueeze, and other shape/data ops."""
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)

    # Truly metadata-only ops incur zero DRAM cost
    zero_cost = node.op_type in _ZERO_COST_OPS

    # For Gather/Slice, actual read is proportional to output (not full input)
    activation_read_elements = None
    if node.op_type in ("Gather", "Slice"):
        out_shape = output_shapes[0] if output_shapes else []
        if out_shape:
            elems = 1
            for d in out_shape:
                elems *= d
            activation_read_elements = elems

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        macs=0,
        is_fused=zero_cost,
        activation_read_elements=activation_read_elements,
    )


def _handle_reduce_mean(node, info: _GraphInfo) -> LayerProfile:
    """Handle ReduceMean — reduction with compute proportional to input size."""
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)
    in_shape = input_shapes[0] if input_shapes else []
    macs = compute_element_wise_macs(in_shape)
    act_elements = _compute_activation_elements(node, info)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        macs=macs,
        activation_read_elements=act_elements,
    )


def _handle_generic(node, info: _GraphInfo) -> LayerProfile:
    """Fallback handler for unrecognized ops."""
    input_shapes = info.get_input_shapes(node)
    output_shapes = info.get_output_shapes(node)
    out_shape = output_shapes[0] if output_shapes else []

    logger.debug("Unhandled op type: %s (node: %s)", node.op_type, node.name)

    return LayerProfile(
        name=node.name or node.output[0],
        op_type=node.op_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        macs=0,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_CONV_OPS = {"Conv", "ConvTranspose"}
_POOL_OPS = {"MaxPool", "AveragePool", "GlobalAveragePool"}
_ACTIVATION_OPS = {"Relu", "LeakyRelu", "Sigmoid", "Clip", "Softmax", "Tanh", "Elu", "HardSigmoid", "HardSwish"}
_ELEMENTWISE_OPS = {"Add", "Sub", "Mul", "Div"}
_ZERO_COST_OPS = {"Reshape", "Flatten", "Squeeze", "Unsqueeze",
                  "Shape", "Constant", "ConstantOfShape"}
_RESHAPE_OPS = {"Reshape", "Flatten", "Transpose", "Squeeze", "Unsqueeze", "Pad",
                "Shape", "Gather", "Slice", "Split", "Cast",
                "Constant", "ConstantOfShape", "Expand", "Tile", "ScatterND"}


def _dispatch_node(
    node,
    info: _GraphInfo,
) -> LayerProfile:
    """Dispatch a node to the appropriate handler."""
    op = node.op_type

    if op in _CONV_OPS:
        return _handle_conv(node, info)
    elif op == "Gemm":
        return _handle_gemm(node, info)
    elif op == "MatMul":
        return _handle_matmul(node, info)
    elif op in _POOL_OPS:
        return _handle_pool(node, info)
    elif op == "BatchNormalization":
        return _handle_batchnorm(node, info)
    elif op in _ACTIVATION_OPS:
        return _handle_activation(node, info)
    elif op in _ELEMENTWISE_OPS:
        return _handle_elementwise(node, info)
    elif op == "Concat":
        return _handle_concat(node, info)
    elif op in ("Resize", "Upsample"):
        return _handle_resize(node, info)
    elif op == "ReduceMean":
        return _handle_reduce_mean(node, info)
    elif op in _RESHAPE_OPS:
        return _handle_reshape_like(node, info)
    else:
        return _handle_generic(node, info)


# ---------------------------------------------------------------------------
# Operator fusion pass
# ---------------------------------------------------------------------------

def _apply_fusion_pass(
    layers: List[LayerProfile],
    graph_nodes: list,
    info: _GraphInfo,
) -> None:
    """
    Post-processing pass to apply common operator fusion patterns.

    Supported fusions:
      - SiLU: Sigmoid(X) + Mul(X, Sigmoid(X)) → single fused activation
      - Conv + Activation: Conv → Relu/Clip fused (activation computed on-chip)
    """
    # Map: tensor name → (graph node, LayerProfile)
    output_to_info = {}
    for node, layer in zip(graph_nodes, layers):
        for out_name in node.output:
            output_to_info[out_name] = (node, layer)

    for node, layer in zip(graph_nodes, layers):
        if layer.is_fused:
            continue

        # --- SiLU fusion: Mul(X, Sigmoid(X)) ---
        if node.op_type == "Mul" and len(node.input) >= 2:
            inp0, inp1 = node.input[0], node.input[1]
            sigmoid_layer = None
            x_name = None

            for sig_candidate, other in [(inp0, inp1), (inp1, inp0)]:
                if sig_candidate in output_to_info:
                    prod_node, prod_layer = output_to_info[sig_candidate]
                    if (prod_node.op_type == "Sigmoid"
                            and len(prod_node.input) >= 1
                            and prod_node.input[0] == other):
                        sigmoid_layer = prod_layer
                        x_name = other
                        break

            if sigmoid_layer is not None:
                # Mark Sigmoid as fused (zero cost)
                sigmoid_layer.is_fused = True
                # Mul represents the full SiLU: sigmoid (4) + multiply (1) = 5 ops/element
                out_shape = layer.primary_output_shape
                out_elems = 1
                for d in out_shape:
                    out_elems *= d
                out_elems = out_elems if out_shape else 0
                layer.ops_override = out_elems * 5
                # Mul only reads x (not also the sigmoid intermediate)
                x_shape = info.get_shape(x_name)
                if x_shape:
                    x_elems = 1
                    for d in x_shape:
                        x_elems *= d
                    layer.activation_read_elements = x_elems
                logger.debug("Fused SiLU: %s + %s", sigmoid_layer.name, layer.name)

        # --- Conv + Activation fusion: activation is free after Conv ---
        elif node.op_type in ("Relu", "Clip") and len(node.input) >= 1:
            inp_name = node.input[0]
            if inp_name in output_to_info:
                prod_node, _ = output_to_info[inp_name]
                if prod_node.op_type in ("Conv", "ConvTranspose"):
                    layer.is_fused = True
                    logger.debug("Fused Conv+%s: %s", node.op_type, layer.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_onnx_model(
    model_path: Union[str, Path],
    input_resolution: Optional[Tuple[int, int]] = None,
) -> List[LayerProfile]:
    """
    Parse an ONNX model file and return a list of LayerProfile objects.

    Args:
        model_path: Path to the .onnx file.
        input_resolution: Optional (H, W) to override the model's input spatial dims.
                          This allows evaluating at different image resolutions.

    Returns:
        List of LayerProfile objects, one per node in the graph.
    """
    model_path = Path(model_path)
    logger.info("Loading ONNX model from %s", model_path)

    model = onnx.load(str(model_path))

    # Override input resolution if specified
    if input_resolution is not None:
        model = _resolve_input_resolution(model, input_resolution)

    # Run shape inference to populate intermediate tensor shapes
    logger.info("Running shape inference...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning("Shape inference failed: %s. Proceeding with available shapes.", e)

    info = _GraphInfo(model.graph)

    layers: List[LayerProfile] = []

    graph_nodes = list(model.graph.node)
    for node in graph_nodes:
        profile = _dispatch_node(node, info)
        layers.append(profile)

    # Apply operator fusion patterns (SiLU, Conv+Activation)
    _apply_fusion_pass(layers, graph_nodes, info)

    # Log summary
    total_macs = sum(l.macs for l in layers)
    total_params = sum(l.weight_params + l.bias_params for l in layers)
    active_layers = sum(1 for l in layers if not l.is_fused and l.macs > 0)
    logger.info(
        "Parsed %d nodes (%d active layers). Total MACs: %.2f G, Total params: %.2f M",
        len(layers),
        active_layers,
        total_macs / 1e9,
        total_params / 1e6,
    )

    return layers
