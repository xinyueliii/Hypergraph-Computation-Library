"""Standalone YOLO11-SoftHGNN object-detection worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _device(value: str):
    if value == "auto":
        return None
    return value.split(":", 1)[1] if value.startswith("cuda:") else value


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    detection_root = source_root / "ObjectDetection"
    sys.path.insert(0, str(detection_root))

    import cv2
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version

    request = json.loads(request_path.read_text(encoding="utf-8"))
    action = request["action"]
    if action not in {"predict", "smoke"}:
        raise ValueError("SoftHGNN object detection supports predict and smoke actions.")
    if not request.get("inputs"):
        raise ValueError("SoftHGNN object detection requires at least one image input.")
    if action == "predict" and not request.get("checkpoint"):
        raise ValueError("A trained SoftHGNN detection checkpoint is required for prediction.")

    parameters = request.get("parameters", {})
    checkpoint = request.get("checkpoint")
    architecture = str(parameters.get("architecture", "yolo11-SoftHGNN"))
    if architecture != "yolo11-SoftHGNN":
        raise ValueError("The shared-runtime integration currently supports yolo11-SoftHGNN only.")
    config = detection_root / "ultralytics" / "cfg" / "models" / "11-SoftHGNN" / "yolo11-SoftHGNN.yaml"
    model_source = Path(checkpoint) if checkpoint else config
    if not model_source.is_file():
        raise FileNotFoundError(f"SoftHGNN model file does not exist: {model_source}")

    seed = int(request.get("seed", 2026))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = YOLO(str(model_source), task="detect")
    run_dir = Path(request["output_dir"]) / request["run_id"]
    rendered_dir = run_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    device = _device(str(request.get("device", "auto")))
    if torch.cuda.is_available() and device not in {"cpu", None}:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results = model.predict(
        source=request["inputs"],
        imgsz=int(parameters.get("image_size", 640)),
        conf=float(parameters.get("confidence", 0.25)),
        iou=float(parameters.get("iou", 0.7)),
        max_det=int(parameters.get("max_detections", 300)),
        device=device,
        save=False,
        verbose=bool(parameters.get("verbose", False)),
    )
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    predictions, artifacts = [], []
    for index, result in enumerate(results):
        boxes = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                result.boxes.cls.detach().cpu().tolist(),
            ):
                class_index = int(class_id)
                boxes.append({
                    "xyxy": xyxy,
                    "confidence": float(confidence),
                    "class_id": class_index,
                    "class_name": result.names[class_index],
                })
        rendered_path = rendered_dir / f"{index:04d}-{Path(result.path).stem}.jpg"
        if not cv2.imwrite(str(rendered_path), result.plot()):
            raise RuntimeError(f"Could not write rendered image: {rendered_path}")
        artifacts.append(str(rendered_path.resolve()))
        predictions.append({"input": str(result.path), "detections": boxes})

    checkpoint_loaded = checkpoint is not None
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "ultralytics": ultralytics_version,
        "architecture": "YOLO11-SoftHGNN-N",
    }
    if torch.cuda.is_available() and device != "cpu":
        environment.update({
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
        })
    warnings = [] if checkpoint_loaded else [
        "No SoftHGNN detection checkpoint is published. This deterministic random-weight run validates architecture, preprocessing and output plumbing only; detections are not meaningful."
    ]
    _write(response_path, {
        "status": "succeeded",
        "predictions": {"items": predictions},
        "artifacts": artifacts,
        "metrics": {"num_inputs": len(predictions), "checkpoint_loaded": checkpoint_loaded},
        "timing": {"worker_inference_seconds": elapsed},
        "environment": environment,
        "warnings": warnings,
        "message": "SoftHGNN detection completed." if checkpoint_loaded else "SoftHGNN detection untrained architecture smoke completed.",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.response, args.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
