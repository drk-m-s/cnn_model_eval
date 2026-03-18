#!/usr/bin/env python3
"""
CNN Model Performance Evaluator — CLI Entry Point.

Evaluates an AI chip's performance running CNN models (ResNet50, YOLOv8, MobileNet)
using a roofline model approach.

Usage examples:
  python main.py --model models/resnet50v1.onnx --resolution 224x224 --quantization int8 --chip-spec chip_specs/example_chips.json
  python main.py --model models/yolov8s.onnx --resolution 640x640 --quantization fp16 --chip-spec chip_specs/example_chips.json --output results/yolov8s_fp16.json
  python main.py --model models/mobilenetv2.onnx --resolution 224x224 --quantization int4 --chip-spec chip_specs/example_chips.json --output results/mobilenet.csv

Supported models (download via models/download_models.py):
  - MobileNetV1, MobileNetV2 (ImageNet, 224x224)
  - ResNet50 V1, ResNet50 V2 (ImageNet, 224x224)
  - YOLOv5s, YOLOv8s         (COCO, 640x640)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.chip import ChipSpec
from src.evaluator import evaluate
from src.export import export_csv, export_json, format_layer_table
from src.onnx_parser import parse_onnx_model


def parse_resolution(s: str) -> tuple[int, int]:
    """Parse a resolution string like '224x224' or '640x480' into (H, W)."""
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{s}'. Expected format: WIDTHxHEIGHT (e.g., 224x224)"
        )
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{s}'. Width and height must be integers."
        )
    return (h, w)  # Return as (H, W) for ONNX convention


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AI chip performance on CNN models using a roofline model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the ONNX model file (e.g., models/resnet50.onnx).",
    )
    parser.add_argument(
        "--resolution", "-r",
        required=True,
        type=parse_resolution,
        help="Input image resolution as WIDTHxHEIGHT (e.g., 224x224, 640x640).",
    )
    parser.add_argument(
        "--quantization", "-q",
        required=True,
        choices=["int8", "fp16", "int4"],
        help="Model quantization type.",
    )
    parser.add_argument(
        "--chip-spec", "-c",
        required=True,
        help="Path to chip specification JSON file.",
    )
    parser.add_argument(
        "--chip-index",
        type=int,
        default=None,
        help="If chip spec JSON contains multiple chips, select by index (0-based). "
             "Default: evaluate all chips.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path for results (.json or .csv). If omitted, results are "
             "only printed to console.",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=None,
        help="Only show the top N layers by execution time in the console table.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logging.error("Model file not found: %s", model_path)
        logging.error(
            "Run 'python models/download_models.py' to download pre-trained ONNX models."
        )
        return 1

    # Load chip specs
    chip_spec_path = Path(args.chip_spec)
    if not chip_spec_path.exists():
        logging.error("Chip spec file not found: %s", chip_spec_path)
        return 1

    chips = ChipSpec.load_all(chip_spec_path)
    if args.chip_index is not None:
        if args.chip_index >= len(chips):
            logging.error(
                "Chip index %d out of range (file has %d chips).",
                args.chip_index,
                len(chips),
            )
            return 1
        chips = [chips[args.chip_index]]

    # Parse ONNX model
    logging.info("Parsing model: %s", model_path.name)
    layers = parse_onnx_model(model_path, input_resolution=args.resolution)

    model_name = model_path.stem

    # Evaluate for each chip
    for chip in chips:
        logging.info("Evaluating on chip: %s", chip.name)

        result = evaluate(
            chip=chip,
            layers=layers,
            quantization=args.quantization,
            input_resolution=args.resolution,
            model_name=model_name,
        )

        # Print summary
        print(result.summary_str())

        # Print layer table
        print()
        print(format_layer_table(result, top_n=args.top_layers))
        print()

        # Export if requested
        if args.output:
            output_path = Path(args.output)
            suffix = output_path.suffix.lower()

            # If multiple chips, append chip name to filename
            if len(chips) > 1:
                stem = output_path.stem + f"_{chip.name.replace(' ', '_')}"
                output_path = output_path.with_stem(stem)

            if suffix == ".json":
                export_json(result, output_path)
                logging.info("Results exported to %s", output_path)
            elif suffix == ".csv":
                export_csv(result, output_path)
                logging.info("Results exported to %s", output_path)
            else:
                # Default to JSON
                output_path = output_path.with_suffix(".json")
                export_json(result, output_path)
                logging.info("Results exported to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
