"""Single processed-mesh inference worker for MeshNet."""

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
    import torch.nn as nn
    import yaml
    from models import MeshNet

    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] not in {"predict", "extract"}:
        raise ValueError("MeshNet supports predict and extract actions.")
    if len(request.get("inputs", [])) != 1:
        raise ValueError("MeshNet requires one processed .npz mesh input.")
    if not request.get("checkpoint"):
        raise ValueError("MeshNet requires a checkpoint path.")
    input_path = Path(request["inputs"][0]).resolve()
    if input_path.suffix.lower() != ".npz":
        raise ValueError("The first MeshNet adapter accepts processed .npz input only.")

    with (source_root / "config" / "test_config.yaml").open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    parameters = request.get("parameters", {})
    max_faces = int(parameters.get("max_faces", configuration["dataset"]["max_faces"]))
    sample = np.load(input_path)
    faces = sample["faces"]
    neighbors = sample["neighbors"]
    if len(faces) > max_faces:
        raise ValueError(f"Mesh has {len(faces)} faces; max_faces is {max_faces}.")
    if len(faces) < max_faces:
        rng = np.random.default_rng(int(request.get("seed", 2026)))
        indices = rng.integers(0, len(faces), size=max_faces - len(faces))
        faces = np.concatenate((faces, faces[indices]), axis=0)
        neighbors = np.concatenate((neighbors, neighbors[indices]), axis=0)

    face_tensor = torch.from_numpy(faces).float().transpose(0, 1).contiguous().unsqueeze(0)
    neighbor_tensor = torch.from_numpy(neighbors).long().unsqueeze(0)
    centers = face_tensor[:, :3]
    corners = face_tensor[:, 3:12] - centers.repeat(1, 3, 1)
    normals = face_tensor[:, 12:]
    requested_device = str(request.get("device", "auto"))
    device = "cuda:0" if requested_device == "auto" else requested_device
    model = MeshNet(cfg=configuration["MeshNet"], require_fea=True)
    if device.startswith("cuda"):
        model = nn.DataParallel(model)
    model = model.to(device)
    checkpoint = torch.load(request["checkpoint"], map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        logits, feature = model(
            centers.to(device), corners.to(device), normals.to(device), neighbor_tensor.to(device))
    probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    predicted_class = int(torch.argmax(logits, dim=1).item())
    output_dir = Path(request["output_dir"]) / request["run_id"]
    feature_path = output_dir / "mesh_feature.npy"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, feature[0].cpu().numpy())
    _write(response_path, {
        "status": "succeeded",
        "predictions": {"class_index": predicted_class, "probabilities": probabilities,
                        "feature": str(feature_path.resolve())},
        "artifacts": [str(feature_path.resolve())],
        "metrics": {},
        "timing": {"worker_inference_seconds": time.perf_counter() - started},
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "message": "MeshNet single-mesh inference completed through HyperComp.",
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
