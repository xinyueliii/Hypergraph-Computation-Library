"""Single-sample inference worker for the internal EvLight V11 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(value, device):
    import torch

    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def _save_image(path: Path, tensor) -> None:
    import numpy as np
    from PIL import Image

    image = tensor.detach().float().clamp(0, 1).cpu()
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    array = image.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB").save(path)


def _ssim(prediction, target) -> float:
    import torch
    import torch.nn.functional as functional

    channels = prediction.shape[1]
    coordinates = torch.arange(11, dtype=prediction.dtype, device=prediction.device) - 5
    gaussian = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] @ gaussian[None, :]).expand(channels, 1, 11, 11)
    mean_prediction = functional.conv2d(prediction, window, padding=5, groups=channels)
    mean_target = functional.conv2d(target, window, padding=5, groups=channels)
    variance_prediction = functional.conv2d(prediction * prediction, window, padding=5, groups=channels) - mean_prediction**2
    variance_target = functional.conv2d(target * target, window, padding=5, groups=channels) - mean_target**2
    covariance = functional.conv2d(prediction * target, window, padding=5, groups=channels) - mean_prediction * mean_target
    numerator = (2 * mean_prediction * mean_target + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_prediction**2 + mean_target**2 + 0.01**2) * (
        variance_prediction + variance_target + 0.03**2
    )
    return float((numerator / denominator).mean().item())


def _model_config(parameters: dict) -> SimpleNamespace:
    base_channels = int(parameters.get("base_channels", 48))
    voxel_channels = int(parameters.get("voxel_grid_channel", 32))
    return SimpleNamespace(
        IlluNet=SimpleNamespace(
            illumiantion_level=1,
            base_chs=base_channels,
            voxel_grid_channel=voxel_channels,
            illumiantion_set=[0],
        ),
        ImageNet=SimpleNamespace(
            snr_factor=3,
            voxel_grid_channel=voxel_channels,
            base_chs=base_channels,
            snr_threshold_list=[0.6, 0.5, 0.4],
        ),
    )


def _prepare_clip_cache(run_dir: Path, parameters: dict) -> None:
    value = parameters.get("clip_checkpoint")
    if not value:
        return
    checkpoint = Path(str(value)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing CLIP checkpoint: {checkpoint}")
    cache_file = run_dir / ".cache" / "clip" / "ViT-B-32.pt"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if not cache_file.exists():
        cache_file.symlink_to(checkpoint)
    os.environ["HOME"] = str(run_dir)


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] not in {"smoke", "predict", "evaluate"}:
        raise ValueError("EvLight V11 supports smoke, predict, and evaluate actions.")
    if len(request.get("inputs", [])) != 1:
        raise ValueError("EvLight V11 requires one SDSD-style sample root.")
    if not request.get("checkpoint"):
        raise ValueError("EvLight V11 requires a trained checkpoint.")

    input_root = Path(request["inputs"][0]).resolve()
    checkpoint = Path(request["checkpoint"]).resolve()
    if not (input_root / "test").is_dir():
        raise FileNotFoundError(f"Missing SDSD-style test split: {input_root / 'test'}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing EvLight V11 checkpoint: {checkpoint}")

    run_dir = Path(request["output_dir"]) / request["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    parameters = dict(request.get("parameters", {}))
    _prepare_clip_cache(run_dir, parameters)

    sys.path.insert(0, str(source_root / "CLIP"))
    sys.path.insert(0, str(source_root / "egllie"))
    sys.path.insert(0, str(source_root))
    import torch
    from torch.utils.data import DataLoader
    from egllie.datasets.egsdsd import get_egsdsd_withNE_dataset
    from egllie.models.egretinex import EgLlie

    requested_device = str(request.get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this environment cannot access a GPU.")

    seed = int(request.get("seed", 2026))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    voxel_channels = int(parameters.get("voxel_grid_channel", 32))
    dataset = get_egsdsd_withNE_dataset(
        dataset_root=str(input_root / "test"),
        center_cropped_height=int(parameters.get("img_height", 260)),
        random_cropped_width=int(parameters.get("img_width", 346)),
        is_train=False,
        is_split_event=True,
        voxel_grid_channel=voxel_channels,
        is_indoor=bool(parameters.get("is_indoor", True)),
    )
    if len(dataset) == 0:
        raise RuntimeError("The EvLight V11 sample root contains no valid test items.")
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    batch = _move(batch, device)

    model = EgLlie(_model_config(parameters)).to(device).eval()
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("state_dict", payload)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unsupported_missing = [
        key for key in missing if not key.startswith("ImageEnhanceNet.clip_model.")
    ]
    if unsupported_missing or unexpected:
        raise RuntimeError(
            "EvLight V11 checkpoint mismatch: "
            f"missing={unsupported_missing}, unexpected={unexpected}"
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    mixed_precision = bool(parameters.get("mixed_precision", True)) and device.type == "cuda"
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            outputs = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    artifacts = {}
    for name, tensor in {
        "input": outputs["input"],
        "prediction": outputs["pred"],
        "ground_truth": outputs["gt"],
        "illumination": outputs["light_up"],
        "blurred_input": batch["lowlight_image_blur"],
    }.items():
        path = run_dir / f"{name}.png"
        _save_image(path, tensor)
        artifacts[name] = str(path.resolve())

    prediction = outputs["pred"].float().clamp(0, 1)
    target = outputs["gt"].float().clamp(0, 1)
    mse = float(torch.mean((prediction - target) ** 2).item())
    metrics = {
        "mse": mse,
        "psnr_db": 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse),
        "ssim": _ssim(prediction, target),
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        environment.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1048576,
            }
        )
    _write(
        response_path,
        {
            "status": "succeeded",
            "predictions": {
                **artifacts,
                "sequence": batch["seq_name"][0],
                "frame_id": batch["frame_id"][0],
                "shape": list(outputs["pred"].shape),
            },
            "artifacts": list(artifacts.values()),
            "metrics": metrics,
            "timing": {"worker_inference_seconds": elapsed},
            "environment": environment,
            "warnings": [
                "The current checkpoint was trained on a 16-sequence SDSD-in fast subset; metrics are not a paper-protocol reproduction."
            ],
            "message": "EvLight V11 checkpoint-backed single-sample inference completed.",
            "checkpoint_sha256": _sha256(checkpoint),
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
