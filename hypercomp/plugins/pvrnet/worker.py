"""Inference-only compatibility worker for the pinned PVRNet source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time


MODELNET40_CLASSES = (
    "night_stand", "range_hood", "plant", "chair", "tent", "curtain", "piano",
    "dresser", "desk", "bed", "sink", "laptop", "flower_pot", "car", "stool",
    "vase", "monitor", "airplane", "stairs", "glass_box", "bottle", "guitar",
    "cone", "toilet", "bathtub", "wardrobe", "radio", "person", "xbox", "bowl",
    "cup", "door", "tv_stand", "mantel", "sofa", "keyboard", "bookshelf",
    "bench", "table", "lamp",
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_state(checkpoint_path: Path, device: str):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        return checkpoint
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    return {key.removeprefix("module."): value for key, value in state.items()}


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] not in {"predict", "extract"}:
        raise ValueError("PVRNet supports predict and extract actions.")
    if len(request.get("inputs", [])) != 2:
        raise ValueError("PVRNet requires a point-cloud .npy file and a directory of 12 views.")
    if not request.get("checkpoint"):
        raise ValueError("PVRNet requires the final PVRNet checkpoint.")
    point_path = Path(request["inputs"][0]).resolve()
    view_root = Path(request["inputs"][1]).resolve()
    if point_path.suffix.lower() != ".npy" or not point_path.is_file():
        raise ValueError("PVRNet's first input must be an existing .npy point cloud.")
    view_paths = sorted(path for path in view_root.iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    expected_views = int(request.get("parameters", {}).get("view_count", 12))
    if len(view_paths) != expected_views:
        raise ValueError(f"PVRNet expected {expected_views} view images, found {len(view_paths)}.")

    sys.path.insert(0, str(source_root))
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from torchvision import transforms

    # Avoid downloading ImageNet weights: the final PVRNet checkpoint restores
    # the complete model. This preserves the upstream architecture only.
    original_alexnet = torchvision.models.alexnet
    torchvision.models.alexnet = lambda *args, **kwargs: original_alexnet(weights=None)
    import config
    config.device = "cuda:0" if request.get("device", "auto") == "auto" else request["device"]
    from datasets import normal_pc
    from models import PVRNet

    points = np.load(point_path)[:1024].astype(np.float32)
    points = normal_pc(points)
    point_tensor = torch.from_numpy(points.T[:, :, None]).float().unsqueeze(0)
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    views = torch.stack([transform(Image.open(path).convert("RGB")) for path in view_paths])
    view_tensor = views.unsqueeze(0)

    # The pinned implementation uses ``x5.squeeze()`` and therefore drops the
    # batch dimension for a one-item batch.  Duplicate the deterministic sample
    # for compatibility, then report only the first prediction/feature.
    point_tensor = point_tensor.repeat(2, 1, 1, 1)
    view_tensor = view_tensor.repeat(2, 1, 1, 1, 1)

    device = config.device
    model = PVRNet(n_classes=40, init_weights=False)
    model.load_state_dict(_load_state(Path(request["checkpoint"]).resolve(), device))
    model = model.to(device).eval()
    point_tensor = point_tensor.to(device)
    view_tensor = view_tensor.to(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        logits, feature = model(point_tensor, view_tensor, get_fea=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    class_index = int(torch.argmax(logits, dim=1)[0].item())
    output_dir = Path(request["output_dir"]) / request["run_id"]
    feature_path = output_dir / "pvrnet_feature.npy"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, feature[0].cpu().numpy())
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "cuda": torch.version.cuda}
    if device.startswith("cuda"):
        environment.update({"gpu": torch.cuda.get_device_name(0),
                            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576})
    _write(response_path, {
        "status": "succeeded",
        "predictions": {"class_index": class_index,
                        "class_name": MODELNET40_CLASSES[class_index],
                        "probabilities": probabilities,
                        "retrieval_feature": str(feature_path.resolve())},
        "artifacts": [str(feature_path.resolve())],
        "metrics": {},
        "timing": {"worker_inference_seconds": elapsed},
        "environment": environment,
        "warnings": ["Inference compatibility is modernized; this is not the original CUDA 9 environment."],
        "message": "PVRNet point-view recognition and feature extraction completed through HyperComp.",
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
