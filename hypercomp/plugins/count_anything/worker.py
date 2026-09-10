"""Single-image text-guided counting worker for Count Anything."""

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


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] != "predict":
        raise ValueError("Count Anything currently supports the predict action only.")
    inputs = request.get("inputs", [])
    if len(inputs) != 1:
        raise ValueError("Count Anything requires exactly one image input.")
    checkpoint = request.get("checkpoint")
    if not checkpoint:
        raise ValueError("Count Anything requires the official count_anything.pt checkpoint.")
    query = str(request.get("parameters", {}).get("query", "")).strip()
    if not query:
        raise ValueError("Count Anything requires --parameter query=<target text>.")
    image_path = Path(inputs[0]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Counting image not found: {image_path}")
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Count Anything checkpoint not found: {checkpoint_path}")

    requested_device = str(request.get("device", "auto"))
    if requested_device not in {"auto", "cuda", "cuda:0"}:
        raise ValueError("The first Count Anything adapter supports one CUDA device only.")
    sys.path.insert(0, str(source_root))
    from count_anything import CountAnything
    import torch

    output_dir = Path(request["output_dir"]) / request["run_id"] / "counting"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = CountAnything(checkpoint_path, output_dir=output_dir, num_gpus=1)
    result = model(image_path, query)[0]
    rendered_path = Path(result.save()).resolve()
    json_path = rendered_path.with_suffix(".json")
    elapsed = time.perf_counter() - started
    points = [dict(item) for item in result.pred_points]
    _write(response_path, {
        "status": "succeeded",
        "predictions": {
            "query": query,
            "count": int(result.count),
            "points": points,
            "visualization": str(rendered_path),
        },
        "artifacts": [str(rendered_path), str(json_path)],
        "metrics": {},
        "timing": {"worker_inference_seconds": elapsed},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
        },
        "message": "Count Anything single-image text-guided counting completed through HyperComp.",
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
