# CNN Model Performance Evaluator

A Python CLI tool that estimates an AI accelerator chip's inference performance (FPS) on CNN models using a **roofline model** approach. It parses ONNX model graphs layer-by-layer, computes per-layer compute (OPS) and memory transfer (DRAM bytes) requirements, then determines whether each layer is **compute-bound** or **memory-bound** against a chip's hardware spec.

## Supported Models

| Model | Task | Default Resolution |
|-------|------|--------------------|
| MobileNet V1 | Classification | 224×224 |
| MobileNet V2 | Classification | 224×224 |
| ResNet50 V1 | Classification | 224×224 |
| ResNet50 V2 | Classification | 224×224 |
| YOLOv5s | Object Detection | 640×640 |
| YOLOv8s | Object Detection | 640×640 |

Any ONNX model can be evaluated — the above are pre-configured for download.

## How It Works

For each layer in the model:

```
compute_time = layer_ops / (effective_TOPS × 1e12)
memory_time  = layer_dram_bytes / (effective_bandwidth × 1e9)
layer_time   = max(compute_time, memory_time)
```

Total inference time is the sum of all layer times (sequential, batch=1). FPS = 1 / total_time.

**Key assumptions:**
- **Roofline model**: each layer is independently bounded by compute or memory
- **No on-chip SRAM caching**: weights + activations are read/written to DRAM each layer (conservative for batch=1)
- **BatchNorm fusion**: BN layers following Conv are fused (zero additional cost)
- **Quantization scaling** (relative to INT8 TOPS):
  - INT8: 1× throughput, 1 byte/element
  - FP16: 0.5× throughput, 2 bytes/element
  - INT4: 2× throughput, 0.5 bytes/element

## Project Structure

```
cnn_model_eval/
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── README.md
├── chip_specs/
│   └── example_chips.json     # Example chip configurations
├── models/
│   ├── download_models.py     # Download/export ONNX models
│   └── *.onnx                 # Model files (after download)
├── src/
│   ├── chip.py                # ChipSpec dataclass
│   ├── layer.py               # LayerProfile dataclass
│   ├── profiler.py            # MAC/DRAM computation helpers
│   ├── onnx_parser.py         # ONNX graph → LayerProfile list
│   ├── evaluator.py           # Roofline model → per-layer time → FPS
│   └── export.py              # JSON/CSV export + console table
├── tests/
│   └── test_evaluator.py      # Unit tests
└── results/                   # Output directory for exports
```

## Setup

```bash
# Create and activate virtual environment
python -m venv cnn_venv
# Windows
cnn_venv\Scripts\Activate.ps1
# Linux/macOS
source cnn_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download ONNX models
python models/download_models.py
```

> **Note:** YOLOv5s and YOLOv8s require the `ultralytics` package (`pip install ultralytics`) for ONNX export. The download script will install it automatically if needed, or print manual instructions.

## Usage

```bash
python main.py --model <onnx_file> --resolution <WxH> --quantization <type> --chip-spec <json_file> [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model, -m` | Path to ONNX model file | `models/resnet50v1.onnx` |
| `--resolution, -r` | Input image resolution (WIDTHxHEIGHT) | `224x224` |
| `--quantization, -q` | Quantization type: `int8`, `fp16`, `int4` | `int8` |
| `--chip-spec, -c` | Path to chip spec JSON file | `chip_specs/example_chips.json` |

### Optional Arguments

| Argument | Description |
|----------|-------------|
| `--chip-index` | Select a specific chip by index (0-based) from the JSON array. Default: evaluate all chips. |
| `--output, -o` | Export results to file (`.json` or `.csv`) |
| `--top-layers` | Only show the top N layers by execution time in the console |
| `--verbose, -v` | Enable verbose logging |

### Examples

```bash
# ResNet50 V1, INT8, on a 10-TOPS NPU
python main.py -m models/resnet50v1.onnx -r 224x224 -q int8 -c chip_specs/example_chips.json --chip-index 1

# YOLOv8s, FP16, on a 30-TOPS NPU, export to JSON
python main.py -m models/yolov8s.onnx -r 640x640 -q fp16 -c chip_specs/example_chips.json --chip-index 2 -o results/yolov8s_fp16.json

python main.py -m models/yolov8s.onnx -r 640x640 -q int8 -c chip_specs/example_chips.json -o results/yolov8s_int8.json

# MobileNetV2, INT4, evaluate on ALL chips in the spec file
python main.py -m models/mobilenetv2.onnx -r 224x224 -q int8 -c chip_specs/example_chips.json

# Show only the top 10 slowest layers
python main.py -m models/resnet50v1.onnx -r 224x224 -q int8 -c chip_specs/example_chips.json --chip-index 0 --top-layers 10
```

## Chip Specification Format

The chip spec JSON file contains an array of chip definitions:

```json
[
  {
    "name": "My_Chip",
    "int8_tops": 10.0,
    "dram_bandwidth_gbps": 25.6,
    "compute_efficiency": 0.70,
    "memory_efficiency": 0.80
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Chip identifier |
| `int8_tops` | float | Peak INT8 throughput in TOPS |
| `dram_bandwidth_gbps` | float | DRAM bandwidth in GB/s |
| `compute_efficiency` | float | Fraction of peak compute achieved (0–1, default: 0.70) |
| `memory_efficiency` | float | Fraction of peak bandwidth achieved (0–1, default: 0.80) |

Four example chips are included in `chip_specs/example_chips.json` (4, 10, 30, 100 TOPS).

## Output

### Console Output

```
======================================================================
  Model       : resnet50v1
  Quantization: INT8
  Resolution  : 224x224
  Chip        : Mid_NPU_10TOPS
    INT8 TOPS : 10.0
    DRAM BW   : 25.6 GB/s
----------------------------------------------------------------------
  Total MACs          :      3.881 G
  Total OPS           :      7.763 G
  Total DRAM traffic  :     76.561 MB
  Total inference time:      3.837 ms
----------------------------------------------------------------------
  >>> FPS (batch=1)   :     260.60 <<<
----------------------------------------------------------------------
  Compute-bound layers: 7
  Memory-bound layers : 115
======================================================================
```

Plus a per-layer table showing MACs, DRAM bytes, compute/memory time, and bottleneck classification.

### JSON Export

Contains full per-layer breakdown plus summary:

```json
{
  "chip": { "name": "...", "int8_tops": 10.0, ... },
  "model": "resnet50v1",
  "quantization": "int8",
  "summary": {
    "total_macs_G": 3.881,
    "total_dram_MB": 76.561,
    "total_inference_time_ms": 3.837,
    "fps": 260.60
  },
  "layers": [ ... ]
}
```

### CSV Export

Tabular per-layer data with a summary footer, suitable for spreadsheet analysis.

## Running Tests

```bash
pip install pytest
python -m pytest tests/test_evaluator.py -v
```

27 unit tests covering MAC computation, DRAM byte estimation, chip spec logic, roofline bottleneck classification, and quantization impact.

## Adding Custom Models

Any ONNX model can be evaluated:

```bash
python main.py --model path/to/custom_model.onnx --resolution 512x512 --quantization int8 --chip-spec chip_specs/example_chips.json
```

The ONNX parser handles Conv, Gemm, MatMul, Pool, BatchNorm (with fusion), activation functions, element-wise ops, Concat, Resize, and reshape-like operations.

## Limitations

- **Batch=1 only**: the model assumes single-image inference with no activation reuse across batches
- **No SRAM modeling**: all data is assumed to transit through DRAM (no on-chip buffer/cache modeling)
- **Sequential execution**: no operator-level parallelism or pipelining is modeled
- **Simplified quantization**: uniform quantization across all layers (no mixed-precision)
- **Theoretical bound**: real-world FPS may differ due to software overhead, scheduling, DMA latency, etc.
