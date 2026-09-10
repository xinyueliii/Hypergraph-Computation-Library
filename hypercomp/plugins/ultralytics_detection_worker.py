"""Standalone Ultralytics-family object-detection worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time


def _device(value: str):
    if value == "auto":
        return None
    if value.startswith("cuda:"):
        return value.split(":", maxsplit=1)[1]
    return value


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    import cv2
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version

    request = json.loads(request_path.read_text(encoding="utf-8"))
    model_name = str(request["component_id"])
    if request["action"] not in {"predict", "smoke"}:
        raise ValueError(f"{model_name} currently supports predict and smoke actions.")
    if not request.get("inputs"):
        raise ValueError(f"{model_name} requires at least one image or video input.")
    if not request.get("checkpoint"):
        raise ValueError(f"{model_name} prediction requires a checkpoint path.")

    parameters = request.get("parameters", {})
    run_dir = Path(request["output_dir"]) / request["run_id"]
    rendered_dir = run_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = YOLO(request["checkpoint"])
    results = model.predict(
        source=request["inputs"],
        imgsz=int(parameters.get("image_size", 640)),
        conf=float(parameters.get("confidence", 0.25)),
        iou=float(parameters.get("iou", 0.7)),
        max_det=int(parameters.get("max_detections", 300)),
        device=_device(request.get("device", "auto")),
        save=False,
        verbose=bool(parameters.get("verbose", False)),
    )

    predictions = []
    artifacts = []
    for index, result in enumerate(results):
        boxes = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                result.boxes.cls.detach().cpu().tolist(),
            ):
                integer_class_id = int(class_id)
                boxes.append(
                    {
                        "xyxy": xyxy,
                        "confidence": float(confidence),
                        "class_id": integer_class_id,
                        "class_name": result.names[integer_class_id],
                    }
                )
        rendered_path = rendered_dir / f"{index:04d}-{Path(result.path).stem}.jpg"
        if not cv2.imwrite(str(rendered_path), result.plot()):
            raise RuntimeError(f"Could not write rendered detection image: {rendered_path}")
        artifacts.append(str(rendered_path.resolve()))
        predictions.append({"input": str(result.path), "detections": boxes})

    _write(
        response_path,
        {
            "status": "succeeded",
            "predictions": {"items": predictions},
            "artifacts": artifacts,
            "metrics": {"num_inputs": len(predictions)},
            "timing": {"worker_seconds": time.perf_counter() - started},
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "ultralytics": ultralytics_version,
            },
            "message": f"{model_name} prediction completed.",
        },
    )


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
