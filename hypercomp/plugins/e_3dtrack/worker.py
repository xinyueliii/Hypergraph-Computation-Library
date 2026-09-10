"""Parameterized remote evaluation worker for E-3DTrack."""

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
    sys.path.insert(0, str(source_root))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from models.tracker_eval import TrackerNetEval
    from utils.dataset import getDataset

    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] not in {"predict", "evaluate"}:
        raise ValueError("E-3DTrack supports predict and evaluate actions.")
    if len(request.get("inputs", [])) != 1:
        raise ValueError("E-3DTrack requires one dataset root containing test.txt.")
    if not request.get("checkpoint"):
        raise ValueError("E-3DTrack requires a checkpoint path.")
    data_root = Path(request["inputs"][0]).resolve()
    if not (data_root / "test.txt").is_file():
        raise FileNotFoundError(f"Missing E-3DTrack test list: {data_root / 'test.txt'}")
    device = request.get("device", "auto")
    device = "cuda:0" if device == "auto" else str(device)
    if not device.startswith("cuda"):
        raise ValueError("The pinned E-3DTrack path requires CUDA.")

    model = TrackerNetEval(feature_dim=384, hgnn=True)
    state = torch.load(request["checkpoint"], map_location=device)
    model.load_state_dict(state["state_dict"])
    model = model.to(device).eval()
    datasets = getDataset(data_folder=str(data_root), train=False)
    output_root = Path(request["output_dir"]) / request["run_id"] / "tracks"
    artifacts: list[str] = []
    started = time.perf_counter()
    with torch.no_grad():
        for sequence in datasets:
            model.reset()
            sequence_output = output_root / sequence.sequence_name
            sequence_output.mkdir(parents=True, exist_ok=True)
            for _timestamp, data, ground_truth in DataLoader(sequence, batch_size=1):
                data = {key: value.to(device) for key, value in data.items()}
                ground_truth = {key: value.to(device) for key, value in ground_truth.items()}
                current_position = data["u_centers_l"]
                prediction = None
                positions_3d = []
                for index in range(data["ev_frame_left"].shape[1]):
                    flow, disparity, prediction = model(
                        data["ev_frame_left"][:, index], data["ev_frame_right"][:, index],
                        data["ref_img"], current_position, None, pred=prediction,
                    )
                    current_position += flow.detach()
                    positions_3d.append(sequence.reprojectImageTo3D_ph(
                        disparity[0].cpu(), current_position[0].cpu()))
                prediction_path = sequence_output / "pos_3d_pred.npy"
                truth_path = sequence_output / "pos_3d_gt.npy"
                np.save(prediction_path, np.asarray(torch.stack(positions_3d).cpu()))
                np.save(truth_path, np.asarray(ground_truth["track_3d"][0].transpose(1, 0).cpu()))
                artifacts.extend([str(prediction_path.resolve()), str(truth_path.resolve())])
    if not artifacts:
        raise RuntimeError("E-3DTrack completed without producing track artifacts.")
    _write(response_path, {
        "status": "succeeded",
        "predictions": {"output_directory": str(output_root.resolve()), "artifact_count": len(artifacts)},
        "artifacts": artifacts,
        "metrics": {},
        "timing": {"worker_inference_seconds": time.perf_counter() - started},
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "message": "E-3DTrack evaluation completed through HyperComp.",
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
