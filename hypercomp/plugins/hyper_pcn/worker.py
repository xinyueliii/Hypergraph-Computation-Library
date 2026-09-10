"""Remote dataset-evaluation worker for the pinned Hyper-PCN repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_direct_inference(request: dict, response_path: Path, source_root: Path) -> None:
    """Load the official checkpoint and complete supplied or generated points."""

    sys.path.insert(0, str(source_root))
    import numpy as np
    import torch
    from models import build_model_from_cfg
    from utils.config import cfg_from_yaml_file

    if not request.get("checkpoint"):
        raise ValueError("Hyper-PCN smoke requires an official checkpoint path.")
    requested_device = str(request.get("device", "auto"))
    device = "cuda:0" if requested_device == "auto" else requested_device
    if not device.startswith("cuda"):
        raise ValueError("The pinned Hyper-PCN implementation requires CUDA extensions.")
    config_path = Path(request.get("parameters", {}).get(
        "config_path", source_root / "cfgs" / "PCN_models" / "Hyper_PCN.yaml")).resolve()
    configuration = cfg_from_yaml_file(str(config_path))
    model = build_model_from_cfg(configuration.model)
    checkpoint = torch.load(request["checkpoint"], map_location="cpu")
    state = checkpoint["base_model"]
    state = {key.replace("module.", "") if "module." in key else key: value
             for key, value in state.items()}
    model.load_state_dict(state)
    model = model.to(device).eval()
    seed = int(request.get("seed", 2026))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = request.get("inputs", [])
    if inputs:
        input_path = Path(inputs[0]).resolve()
        if input_path.suffix.lower() != ".npy" or not input_path.is_file():
            raise ValueError("Hyper-PCN direct inference expects one existing .npy point cloud.")
        input_points = np.load(input_path).astype(np.float32)
        if input_points.ndim == 3 and input_points.shape[0] == 1:
            input_points = input_points[0]
        if input_points.ndim != 2 or input_points.shape[1] != 3:
            raise ValueError("Hyper-PCN input must have shape [N, 3] or [1, N, 3].")
        points = torch.from_numpy(input_points).unsqueeze(0).to(device)
        input_kind = "provided_npy"
    else:
        point_count = int(request.get("parameters", {}).get("point_count", 2048))
        points = torch.rand((1, point_count, 3), device=device) * 2 - 1
        input_kind = "synthetic_random"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(points)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    run_dir = Path(request["output_dir"]) / request["run_id"]
    output_path = run_dir / "completed_point_cloud.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = outputs[-1][0].detach().cpu().numpy()
    partial = points[0].detach().cpu().numpy()
    np.save(output_path, completed)
    visualization_path = run_dir / "completion_visualization.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(8, 4), dpi=150)
    for index, (cloud, title) in enumerate(((partial, "Partial input"), (completed, "Hyper-PCN completion")), 1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        stride = max(1, len(cloud) // 4096)
        shown = cloud[::stride]
        axis.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=0.5, c=shown[:, 2], cmap="viridis")
        axis.set_title(title)
        axis.set_axis_off()
        axis.set_box_aspect((1, 1, 1))
    figure.tight_layout()
    figure.savefig(visualization_path, bbox_inches="tight")
    plt.close(figure)
    _write(response_path, {
        "status": "succeeded",
        "predictions": {
            "completed_point_cloud": str(output_path.resolve()),
            "visualization": str(visualization_path.resolve()),
            "input_kind": input_kind,
            "stage_shapes": [list(output.shape) for output in outputs],
        },
        "artifacts": [str(output_path.resolve()), str(visualization_path.resolve())],
        "metrics": {},
        "timing": {"worker_inference_seconds": elapsed},
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
        },
        "warnings": (["Synthetic random points validate the adapter, not PCN benchmark accuracy."]
                     if input_kind == "synthetic_random" else
                     ["The mesh-derived partial sample validates inference, not PCN benchmark accuracy."]),
        "message": "Hyper-PCN official-checkpoint direct inference completed.",
    })


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] == "smoke" or (request["action"] == "predict" and request.get("inputs")):
        _run_direct_inference(request, response_path, source_root)
        return
    if request["action"] not in {"predict", "evaluate"}:
        raise ValueError("Hyper-PCN supports smoke, predict, and evaluate actions.")
    if not request.get("checkpoint"):
        raise ValueError("Hyper-PCN requires a checkpoint path.")
    parameters = request.get("parameters", {})
    config_path = Path(parameters.get(
        "config_path", source_root / "cfgs" / "PCN_models" / "Hyper_PCN.yaml")).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hyper-PCN config: {config_path}")
    device = str(request.get("device", "auto"))
    if device != "auto" and not device.startswith("cuda"):
        raise ValueError("The pinned Hyper-PCN implementation requires CUDA extensions.")
    output_dir = Path(request["output_dir"]) / request["run_id"] / "point_clouds"
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if ":" in device:
        environment["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    command = [
        sys.executable, str(source_root / "main.py"), "--test",
        "--ckpts", str(Path(request["checkpoint"]).resolve()),
        "--config", str(config_path), "--num_workers", str(int(parameters.get("num_workers", 2))),
        "--exp_name", request["run_id"], "--dump_for_kitti_metric",
        "--dump_root", str(output_dir.resolve()),
    ]
    if parameters.get("dump_points"):
        command.extend(["--dump_points", str(int(parameters["dump_points"]))])
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=source_root, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Hyper-PCN upstream evaluation exited with {completed.returncode}.")
    artifacts = sorted(str(path.resolve()) for path in output_dir.rglob("*.npy"))
    _write(response_path, {
        "status": "succeeded",
        "predictions": {"output_directory": str(output_dir.resolve()), "point_cloud_count": len(artifacts)},
        "artifacts": artifacts,
        "metrics": {},
        "timing": {"worker_evaluation_seconds": time.perf_counter() - started},
        "environment": {"python": platform.python_version()},
        "warnings": ["Metrics remain in upstream stdout until a stable parser is verified."],
        "message": "Hyper-PCN dataset evaluation completed through HyperComp.",
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
