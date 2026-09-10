"""Standalone SoftHGNN image-classification inference worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time


_CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this environment cannot access a GPU.")
    return device


def _checkpoint_state(torch, checkpoint_path: Path, device) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SoftHGNN checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError("SoftHGNN checkpoint must contain a state-dict mapping.")
    for key in ("model", "net", "state_dict", "model_state"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            payload = candidate
            break
    return {
        (key[7:] if isinstance(key, str) and key.startswith("module.") else key): value
        for key, value in payload.items()
    }


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    classification_root = source_root / "ImageClassification"
    sys.path.insert(0, str(classification_root))

    import einops
    import torch
    from PIL import Image
    from torchvision import transforms
    from models.vit_softhgnn import ViTWithSoftHGNN

    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] not in {"predict", "smoke"}:
        raise ValueError("SoftHGNN image classification currently supports predict and smoke actions.")
    if not request.get("inputs"):
        raise ValueError("SoftHGNN requires at least one image input.")
    checkpoint = request.get("checkpoint")
    if request["action"] == "predict" and not checkpoint:
        raise ValueError("SoftHGNN prediction requires a trained checkpoint.")

    parameters = request.get("parameters", {})
    dataset = str(parameters.get("dataset", "cifar10")).lower()
    if dataset not in {"cifar10", "cifar100"}:
        raise ValueError("SoftHGNN dataset must be 'cifar10' or 'cifar100'.")
    num_classes = 10 if dataset == "cifar10" else 100
    image_size = int(parameters.get("image_size", 32))
    patch_size = int(parameters.get("patch_size", 4))
    top_k = min(max(int(parameters.get("top_k", 5)), 1), num_classes)
    device = _device(torch, str(request.get("device", "auto")))
    seed = int(request.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ViTWithSoftHGNN(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=int(parameters.get("embedding_dim", 512)),
        depth=int(parameters.get("depth", 6)),
        heads=int(parameters.get("attention_heads", 8)),
        dim_head=int(parameters.get("attention_head_dim", 64)),
        mlp_dim=int(parameters.get("mlp_dim", 512)),
        num_hyperedges=int(parameters.get("num_hyperedges", 16)),
        num_edge_head=int(parameters.get("hyperedge_heads", 8)),
        pool=str(parameters.get("pool", "cls")),
        channels=3,
        dropout=float(parameters.get("dropout", 0.1)),
        emb_dropout=float(parameters.get("embedding_dropout", 0.1)),
    ).to(device)
    warnings = []
    checkpoint_loaded = checkpoint is not None
    if checkpoint_loaded:
        model.load_state_dict(_checkpoint_state(torch, Path(checkpoint), device))
    else:
        warnings.append(
            "SoftHGNN smoke run used deterministic randomly initialized weights; "
            "its class scores are an integration check, not a trained prediction."
        )
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    input_paths = [Path(value) for value in request["inputs"]]
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(f"SoftHGNN image input does not exist: {input_path}")
    batch = torch.stack(
        [preprocess(Image.open(path).convert("RGB")) for path in input_paths]
    ).to(device)

    started = time.perf_counter()
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)
    top_probabilities, top_indices = probabilities.topk(top_k, dim=1)
    worker_seconds = time.perf_counter() - started

    configured_names = parameters.get("class_names")
    if configured_names is not None:
        if not isinstance(configured_names, list) or len(configured_names) != num_classes:
            raise ValueError(f"class_names must be a JSON list with exactly {num_classes} entries.")
    class_names = configured_names or (_CIFAR10_CLASSES if dataset == "cifar10" else None)
    predictions = []
    for path, indices, scores in zip(input_paths, top_indices.cpu(), top_probabilities.cpu()):
        candidates = []
        for class_id, score in zip(indices.tolist(), scores.tolist()):
            candidates.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_names[class_id] if class_names is not None else None,
                    "probability": float(score),
                }
            )
        predictions.append({"input": str(path), "top_k": candidates})

    _write(
        response_path,
        {
            "status": "succeeded",
            "predictions": {"dataset": dataset, "items": predictions},
            "artifacts": [],
            "metrics": {
                "num_inputs": len(predictions),
                "checkpoint_loaded": checkpoint_loaded,
            },
            "timing": {"worker_inference_seconds": worker_seconds},
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "einops": einops.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
            },
            "warnings": warnings,
            "message": (
                "SoftHGNN image classification completed."
                if checkpoint_loaded
                else "SoftHGNN untrained smoke forward completed."
            ),
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
