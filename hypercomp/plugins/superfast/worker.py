"""Standalone single-sample SuperFast worker for the THU-HSEVI layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    import imageio
    import numpy as np
    import torch
    from torch import nn
    from model import FusionModel
    import test_THU_HSEVI as upstream_test

    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] not in {"predict", "smoke"}:
        raise ValueError("SuperFast worker currently supports predict and smoke actions.")
    if len(request.get("inputs", [])) != 1:
        raise ValueError("SuperFast requires one THU-HSEVI scene directory as input.")
    if not request.get("checkpoint"):
        raise ValueError("SuperFast prediction requires a checkpoint path.")
    requested_device = request.get("device", "auto")
    device = "cuda:0" if requested_device == "auto" else requested_device
    if not str(device).startswith("cuda"):
        raise ValueError("The upstream SuperFast/slayerSNN path currently requires CUDA.")
    device_parts = str(device).split(":", maxsplit=1)
    cuda_index = int(device_parts[1]) if len(device_parts) == 2 else 0
    torch.cuda.set_device(cuda_index)

    parameters = request.get("parameters", {})
    sample_index = int(parameters.get("sample_index", 0))
    number_of_time_bins = int(parameters.get("number_of_time_bins", 15))
    scene_directory = Path(request["inputs"][0]).resolve()
    run_dir = Path(request["output_dir"]) / request["run_id"]
    output_path = run_dir / "interpolated.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The upstream class references os imported only in its script main block.
    upstream_test.os = os
    dataset = upstream_test.UHSEDataset(str(scene_directory), nb_of_timebin=number_of_time_bins)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {sample_index} is outside the valid range 0..{len(dataset) - 1}."
        )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [sample_index]),
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )
    sample = next(iter(loader))
    (
        events_forward,
        events_backward,
        left_image,
        right_image,
        _ground_truth,
        weight,
        n_left,
        n_right,
        surface,
        left_voxel_grid,
        right_voxel_grid,
        source_index,
    ) = sample

    model_parameters = {"Ts": 1, "tSample": number_of_time_bins * 2}
    model = FusionModel(
        model_parameters,
        hidden_number=32,
        theta=[3, 5, 10],
        tauSr=[1, 2, 4],
        tauRef=[1, 2, 4],
        scaleRef=[1, 1, 1],
        tauRho=[1, 1, 10],
        scaleRho=[1, 1, 10],
        channel=1,
    )
    model = nn.DataParallel(model, device_ids=[cuda_index], output_device=cuda_index).to(device)
    checkpoint = torch.load(request["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    tensors = [
        events_forward,
        events_backward,
        left_image,
        right_image,
        weight,
        surface,
        left_voxel_grid,
        right_voxel_grid,
    ]
    tensors = [value.to(device) for value in tensors]
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        output = model(
            tensors[0],
            tensors[1],
            tensors[2],
            tensors[3],
            tensors[4],
            n_left,
            n_right,
            tensors[5],
            tensors[6],
            tensors[7],
        )
    output_image = output[0, 0].detach().clamp(0, 1).cpu().numpy()
    imageio.imwrite(output_path, (output_image * 255).astype(np.uint8))

    _write(
        response_path,
        {
            "status": "succeeded",
            "predictions": {
                "generated_frame": str(output_path.resolve()),
                "sample_index": int(source_index[0]),
                "resolution": [int(output_image.shape[1]), int(output_image.shape[0])],
            },
            "artifacts": [str(output_path.resolve())],
            "metrics": {},
            "timing": {"worker_inference_seconds": time.perf_counter() - started},
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "message": "SuperFast single-sample interpolation completed.",
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
