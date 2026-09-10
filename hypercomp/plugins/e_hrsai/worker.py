"""Remote E-HRSAI worker that parameterizes the upstream evaluation command."""

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


def _run_synthetic_smoke(request: dict, response_path: Path, source_root: Path) -> None:
    """Run the tensor layout documented by the upstream model source."""

    sys.path.insert(0, str(source_root))
    import cv2
    import numpy as np
    import torch
    from model.EDeOccSR import EDeOccSR

    if not request.get("checkpoint"):
        raise ValueError("E-HRSAI smoke requires an official checkpoint path.")
    requested_device = str(request.get("device", "auto"))
    device = "cuda:0" if requested_device == "auto" else requested_device
    if not device.startswith("cuda"):
        raise ValueError("The pinned E-HRSAI implementation requires CUDA.")
    torch.manual_seed(int(request.get("seed", 2026)))
    torch.cuda.manual_seed_all(int(request.get("seed", 2026)))
    model = EDeOccSR(
        input_img_channel=51, input_event_channel=36, dim=48,
        num_blocks=[1, 2, 2, 3], num_refinement_blocks=2,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        LayerNorm_type="WithBias", bias=False, img_channel=3,
        channel_att=True, rate=4,
    ).to(device).eval()
    state = torch.load(request["checkpoint"], map_location="cpu")
    model.load_state_dict(state["state_dict"])
    zeros = lambda shape: torch.zeros(shape, device=device)
    inputs = (
        zeros((1, 3, 51, 256, 256)), zeros((1, 3, 17, 256, 256)),
        zeros((1, 3, 36, 256, 256)), zeros((1, 5, 256, 256)),
        zeros((1, 5, 256, 256)),
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        output = model(*inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    run_dir = Path(request["output_dir"]) / request["run_id"]
    output_path = run_dir / "synthetic_reconstruction.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = output[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(output_path), cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    _write(response_path, {
        "status": "succeeded",
        "predictions": {"reconstruction": str(output_path.resolve()), "shape": list(output.shape)},
        "artifacts": [str(output_path.resolve())],
        "metrics": {},
        "timing": {"worker_inference_seconds": elapsed},
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
        },
        "warnings": ["Synthetic zero tensors validate the adapter, not official-dataset accuracy."],
        "message": "E-HRSAI official-checkpoint synthetic smoke completed.",
    })


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] == "smoke":
        _run_synthetic_smoke(request, response_path, source_root)
        return
    if request["action"] not in {"predict", "evaluate"}:
        raise ValueError("E-HRSAI supports smoke, predict, and evaluate actions.")
    if len(request.get("inputs", [])) != 1:
        raise ValueError("E-HRSAI requires one dataset root containing test.txt.")
    if not request.get("checkpoint"):
        raise ValueError("E-HRSAI requires an official checkpoint path.")

    dataset_root = Path(request["inputs"][0]).resolve()
    if not (dataset_root / "test.txt").is_file():
        raise FileNotFoundError(f"Missing E-HRSAI test list: {dataset_root / 'test.txt'}")
    output_dir = Path(request["output_dir"]) / request["run_id"] / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = request.get("device", "auto")
    if device == "auto":
        device = "cuda"
    if not str(device).startswith("cuda"):
        raise ValueError("The pinned E-HRSAI implementation requires CUDA.")

    environment = os.environ.copy()
    if ":" in str(device):
        environment["CUDA_VISIBLE_DEVICES"] = str(device).split(":", 1)[1]
    command = [
        sys.executable,
        str(source_root / "eval.py"),
        "--folder", str(dataset_root),
        "--ckpt", str(Path(request["checkpoint"]).resolve()),
        "--opFolder", str(output_dir.resolve()),
        "--device", "cuda",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=source_root, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"E-HRSAI upstream evaluation exited with {completed.returncode}.")
    artifacts = sorted(str(path.resolve()) for path in output_dir.glob("*.png"))
    if not artifacts:
        raise RuntimeError("E-HRSAI completed without producing PNG artifacts.")
    _write(response_path, {
        "status": "succeeded",
        "predictions": {"output_directory": str(output_dir.resolve()), "image_count": len(artifacts)},
        "artifacts": artifacts,
        "metrics": {},
        "timing": {"worker_inference_seconds": time.perf_counter() - started},
        "environment": {"python": platform.python_version()},
        "message": "E-HRSAI upstream evaluation completed through HyperComp.",
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
