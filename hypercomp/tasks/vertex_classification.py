"""Shared native vertex-classification task for HGNN-family models."""

from __future__ import annotations

import platform
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

import dhg
from dhg.models import HGNN, HGNNP

from hypercomp.contracts import TaskRequest, TaskResult
from hypercomp.serialization import write_json


_MODEL_TYPES = {
    "hgnn": HGNN,
    "hgnnp": HGNNP,
}


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment cannot access a GPU.")
    return device


def _build_tiny_dataset(device: torch.device, seed: int) -> dict[str, Any]:
    """Build a deterministic, redistributable smoke dataset with three classes."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4, dtype=torch.long)
    features = torch.zeros(12, 6)
    features[range(12), labels] = 1.0
    features[:, 3:] = torch.randn(12, 3, generator=generator) * 0.03
    hyperedges = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (2, 3, 4, 5),
        (6, 7, 8, 9),
    ]
    train_mask = torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.long)
    val_mask = torch.tensor([2, 6, 10], dtype=torch.long)
    test_mask = torch.tensor([3, 7, 11], dtype=torch.long)
    return {
        "features": features.to(device),
        "labels": labels.to(device),
        "hypergraph": dhg.Hypergraph(12, hyperedges).to(device),
        "train_mask": train_mask.to(device),
        "val_mask": val_mask.to(device),
        "test_mask": test_mask.to(device),
        "num_classes": 3,
    }


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def _environment(device: torch.device) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": gpu_name,
        "dhg": getattr(dhg, "__version__", "0.9.7-source"),
    }


def _read_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch 1.x does not expose the safer weights_only option.
        return torch.load(checkpoint_path, map_location=device)


def run_vertex_classification(request: TaskRequest) -> TaskResult:
    """Train, evaluate, predict, or smoke-test HGNN/HGNN+ on a fixed sample."""

    if request.component_id not in _MODEL_TYPES:
        raise KeyError(f"Unsupported native vertex-classification model: {request.component_id}")
    if request.action not in {"smoke", "train", "evaluate", "predict"}:
        raise ValueError(f"Unsupported action: {request.action}")

    started = time.perf_counter()
    torch.manual_seed(request.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(request.seed)

    device = _resolve_device(request.device)
    dataset = _build_tiny_dataset(device, request.seed)
    checkpoint_payload = None
    if request.checkpoint is not None:
        checkpoint_payload = _read_checkpoint(request.checkpoint, device)
        checkpoint_component = checkpoint_payload.get("component_id")
        if checkpoint_component != request.component_id:
            raise ValueError(
                f"Checkpoint component '{checkpoint_component}' does not match "
                f"requested component '{request.component_id}'."
            )
    hidden_dim = int(
        request.parameters.get(
            "hidden_dim",
            checkpoint_payload.get("hidden_dim", 8) if checkpoint_payload else 8,
        )
    )
    drop_rate = float(
        request.parameters.get(
            "drop_rate",
            checkpoint_payload.get("drop_rate", 0.0) if checkpoint_payload else 0.0,
        )
    )
    model = _MODEL_TYPES[request.component_id](
        dataset["features"].shape[1],
        hidden_dim,
        dataset["num_classes"],
        drop_rate=drop_rate,
    ).to(device)

    run_dir = request.output_dir / request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = write_json(run_dir / "request.json", request)
    environment = _environment(device)
    environment_path = write_json(run_dir / "environment.json", environment)
    checkpoint_path = request.checkpoint
    last_loss = 0.0

    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model_state"])

    if request.action in {"smoke", "train"}:
        epochs = int(request.parameters.get("epochs", 5 if request.action == "smoke" else 50))
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(request.parameters.get("learning_rate", 0.03)),
            weight_decay=float(request.parameters.get("weight_decay", 5e-4)),
        )
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(dataset["features"], dataset["hypergraph"])
            loss = F.cross_entropy(
                logits[dataset["train_mask"]],
                dataset["labels"][dataset["train_mask"]],
            )
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        checkpoint_path = run_dir / "checkpoint.pt"
        torch.save(
            {
                "component_id": request.component_id,
                "model_state": model.state_dict(),
                "hidden_dim": hidden_dim,
                "drop_rate": drop_rate,
                "seed": request.seed,
            },
            checkpoint_path,
        )
    elif checkpoint_path is None:
        raise ValueError(f"Action '{request.action}' requires --checkpoint.")

    model.eval()
    with torch.no_grad():
        logits = model(dataset["features"], dataset["hypergraph"])
    predictions = logits.argmax(dim=1).detach().cpu().tolist()
    metrics = {
        "loss": last_loss,
        "validation_accuracy": _accuracy(
            logits[dataset["val_mask"]], dataset["labels"][dataset["val_mask"]]
        ),
        "test_accuracy": _accuracy(
            logits[dataset["test_mask"]], dataset["labels"][dataset["test_mask"]]
        ),
    }
    predictions_path = write_json(run_dir / "predictions.json", {"classes": predictions})
    metrics_path = write_json(run_dir / "metrics.json", metrics)
    elapsed = time.perf_counter() - started
    artifacts = [request_path, environment_path, predictions_path, metrics_path]
    if checkpoint_path is not None:
        artifacts.append(checkpoint_path)

    result_path = run_dir / "result.json"
    result = TaskResult(
        run_id=request.run_id,
        component_id=request.component_id,
        status="succeeded",
        task_id=request.task_id or "vertex_classification",
        predictions={"classes": predictions},
        artifacts=tuple(artifacts) + (result_path,),
        metrics=metrics,
        timing={"total_seconds": elapsed},
        environment=environment,
        message=f"{request.component_id} {request.action} completed on {device}.",
    )
    write_json(result_path, result)
    return result
