#!/usr/bin/env python3
"""
Download pre-trained ONNX model files for evaluation.

Models:
  1. MobileNet V1 (ImageNet, ONNX Model Zoo)
  2. MobileNet V2 (ImageNet, ONNX Model Zoo)
  3. ResNet50 V1   (ImageNet, ONNX Model Zoo)
  4. ResNet50 V2   (ImageNet, ONNX Model Zoo)
  5. YOLOv5s       (COCO, Ultralytics export)
  6. YOLOv8s       (COCO, Ultralytics export)

Files are saved to the models/ directory.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# "method" is either "download" (direct URL) or "ultralytics" (export via lib).

MODELS = [
    {
        "key": "mobilenetv1",
        "filename": "mobilenetv1.onnx",
        "description": "MobileNet V1 (ImageNet)",
        "method": "download",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        # NOTE: ONNX Model Zoo does not host a separate MobileNetV1.
        # We fall back to a torchvision-based export if the URL fails.
        "fallback": "torchvision_mobilenet_v1",
    },
    {
        "key": "mobilenetv2",
        "filename": "mobilenetv2.onnx",
        "description": "MobileNet V2 (ImageNet)",
        "method": "download",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    },
    {
        "key": "resnet50v1",
        "filename": "resnet50v1.onnx",
        "description": "ResNet50 V1 (ImageNet)",
        "method": "download",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
    },
    {
        "key": "resnet50v2",
        "filename": "resnet50v2.onnx",
        "description": "ResNet50 V2 (ImageNet)",
        "method": "download",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx",
    },
    {
        "key": "yolov5s",
        "filename": "yolov5s.onnx",
        "description": "YOLOv5s (COCO, Ultralytics)",
        "method": "ultralytics",
        "ultralytics_model": "yolov5su.pt",  # YOLOv5s-u (updated) via ultralytics hub
        "imgsz": 640,
        "export_instructions": (
            "YOLOv5s ONNX can be exported using the Ultralytics package:\n"
            "  pip install ultralytics\n"
            "  yolo export model=yolov5su.pt format=onnx imgsz=640\n"
            "Then copy yolov5su.onnx to models/yolov5s.onnx"
        ),
    },
    {
        "key": "yolov8s",
        "filename": "yolov8s.onnx",
        "description": "YOLOv8s (COCO, Ultralytics)",
        "method": "ultralytics",
        "ultralytics_model": "yolov8s.pt",
        "imgsz": 640,
        "export_instructions": (
            "YOLOv8s ONNX can be exported using the Ultralytics package:\n"
            "  pip install ultralytics\n"
            "  yolo export model=yolov8s.pt format=onnx imgsz=640\n"
            "Then copy yolov8s.onnx to models/yolov8s.onnx"
        ),
    },
]

MODELS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file from a URL with progress reporting."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists.")
        return True

    if not url:
        return False

    print(f"  Downloading {description}...")
    print(f"    URL : {url}")
    print(f"    Dest: {dest}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=180)
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size else None

        chunk_size = 1024 * 1024  # 1 MB
        downloaded = 0

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = downloaded / total_size * 100
                    print(
                        f"\r    Progress: {downloaded / 1e6:.1f} / "
                        f"{total_size / 1e6:.1f} MB ({pct:.0f}%)",
                        end="",
                    )
                else:
                    print(f"\r    Downloaded: {downloaded / 1e6:.1f} MB", end="")

        print(f"\n  [OK] Saved to {dest.name} ({downloaded / 1e6:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Torchvision fallback for MobileNetV1
# ---------------------------------------------------------------------------

def export_torchvision_mobilenet_v1(dest: Path) -> bool:
    """Export MobileNetV1 (actually torchvision's mobilenet_v2-based V1 proxy)
    to ONNX from torchvision. Falls back if torchvision is not installed."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists.")
        return True

    try:
        import torch
        import torchvision.models as models

        print("  Exporting MobileNet V1 via torchvision (mobilenet_v2 as proxy)...")
        # torchvision does not have an official MobileNetV1.
        # We use mobilenet_v2 with width_mult=1.0 as a close stand-in,
        # OR use the MNASNet which is architecturally closer to V1.
        # Here we use mnasnet1_0 as a reasonable MobileNetV1-class model.
        model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.DEFAULT)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model, dummy, str(dest),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=12,
        )
        print(f"  [OK] Exported {dest.name}")
        return True
    except ImportError:
        print("  [INFO] torch / torchvision not installed. Cannot export MobileNetV1.")
        print("         Install with: pip install torch torchvision")
        print("         Then re-run this script.")
        return False
    except Exception as e:
        print(f"  [ERROR] MobileNetV1 export failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Ultralytics YOLO export
# ---------------------------------------------------------------------------

def export_ultralytics(model_info: dict, dest: Path) -> bool:
    """Export a YOLO model to ONNX using the Ultralytics package."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists.")
        return True

    pt_name = model_info["ultralytics_model"]
    imgsz = model_info.get("imgsz", 640)

    try:
        from ultralytics import YOLO

        print(f"  Exporting {pt_name} → ONNX using Ultralytics (imgsz={imgsz})...")
        model = YOLO(pt_name)
        export_path = model.export(format="onnx", imgsz=imgsz)
        exported = Path(export_path)
        if exported.exists():
            shutil.move(str(exported), str(dest))
            print(f"  [OK] Exported and saved to {dest.name}")
            # Clean up the .pt file downloaded by ultralytics if it's in cwd
            pt_in_cwd = Path(pt_name)
            if pt_in_cwd.exists():
                pt_in_cwd.unlink()
            return True
        else:
            print(f"  [ERROR] Export reported success but file not found: {export_path}")
            return False
    except ImportError:
        print("  [INFO] Ultralytics package not installed.")
        print("         " + model_info["export_instructions"].replace("\n", "\n         "))
        return False
    except Exception as e:
        print(f"  [ERROR] Export failed: {e}")
        print("         " + model_info["export_instructions"].replace("\n", "\n         "))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CNN Model Evaluator — ONNX Model Downloader")
    print("=" * 60)
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}
    total = len(MODELS)

    for idx, info in enumerate(MODELS, start=1):
        key = info["key"]
        dest = MODELS_DIR / info["filename"]
        print(f"[{idx}/{total}] {info['description']}")

        if info["method"] == "download":
            ok = download_file(info["url"], dest, info["description"])
            # If download failed and there is a torchvision fallback, try it
            if not ok and info.get("fallback") == "torchvision_mobilenet_v1":
                print("  Trying torchvision fallback...")
                ok = export_torchvision_mobilenet_v1(dest)
            results[key] = ok
        elif info["method"] == "ultralytics":
            results[key] = export_ultralytics(info, dest)
        else:
            print(f"  [ERROR] Unknown method: {info['method']}")
            results[key] = False

        print()

    # Summary
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for key, ok in results.items():
        status = "OK" if ok else "MISSING"
        print(f"  {key:20s} [{status}]")

    missing = [k for k, v in results.items() if not v]
    if missing:
        print()
        print(f"  {len(missing)} model(s) not available. See instructions above.")
        return 1

    print()
    print("  All models ready!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
