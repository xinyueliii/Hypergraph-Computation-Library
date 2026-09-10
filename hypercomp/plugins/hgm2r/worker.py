"""Standalone HGM2R multimodal 3D retrieval integration worker."""

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


def _device(torch, value: str):
    selected = "cuda:0" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this environment cannot access a GPU.")
    return device


def _load_modalities(np, paths: list[Path]):
    payloads = [np.load(path, allow_pickle=True).item() for path in paths]
    required = {"feature", "label", "train_idx", "query_idx", "target_idx"}
    for path, payload in zip(paths, payloads):
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"HGM2R feature file {path} is missing keys: {sorted(missing)}")
    count = len(payloads[0]["label"])
    reference = payloads[0]
    for path, payload in zip(paths[1:], payloads[1:]):
        if len(payload["label"]) != count or not np.array_equal(payload["label"], reference["label"]):
            raise ValueError(f"HGM2R modalities are not sample/label aligned: {path}")
        for key in ("train_idx", "query_idx", "target_idx"):
            if not np.array_equal(payload[key], reference[key]):
                raise ValueError(f"HGM2R modalities use different {key}: {path}")
    return payloads


def run(request_path: Path, response_path: Path, source_root: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["action"] != "smoke":
        raise ValueError("HGM2R currently supports smoke only because upstream publishes no trained checkpoint.")
    if len(request.get("inputs", [])) != 3:
        raise ValueError("HGM2R requires three aligned .npy inputs: point-cloud, voxel and multi-view features.")
    paths = [Path(value) for value in request["inputs"]]
    if any(path.suffix.lower() != ".npy" or not path.is_file() for path in paths):
        raise ValueError("All HGM2R inputs must be existing .npy feature files.")

    sys.path.insert(0, str(source_root))
    import numpy as np
    import scipy
    import torch
    import torch.nn as nn
    from models import CMAE, HGNN

    parameters = request.get("parameters", {})
    device = _device(torch, str(request.get("device", "auto")))
    seed = int(request.get("seed", 2026))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    payloads = _load_modalities(np, paths)
    features = [torch.from_numpy(item["feature"]).float().to(device) for item in payloads]
    labels = torch.from_numpy(np.asarray(payloads[0]["label"])).long().to(device)
    train_mask = torch.from_numpy(np.asarray(payloads[0]["train_idx"]).astype(bool)).to(device)
    query_mask = np.asarray(payloads[0]["query_idx"]).astype(bool)
    target_mask = np.asarray(payloads[0]["target_idx"]).astype(bool)

    # One optimization step exercises the paper's cross-modal alignment path on
    # the bundled ESB features without pretending to reproduce a trained model.
    cmae = CMAE([item.shape[1] for item in features]).to(device)
    cmae.train()
    optimizer = torch.optim.SGD(cmae.parameters(), lr=float(parameters.get("learning_rate", 0.01)), momentum=0.9)
    optimizer.zero_grad()
    xs, codes, reconstructed, cross_reconstructed = cmae(features)
    pair_losses = [nn.functional.mse_loss(codes[i], codes[j]) for i in range(3) for j in range(i + 1, 3)]
    reconstruction_losses = [nn.functional.mse_loss(xs[i], reconstructed[i]) + nn.functional.mse_loss(xs[i], cross_reconstructed[i]) for i in range(3)]
    cmae_loss = 0.6 * torch.stack(pair_losses).mean() + 0.2 * torch.stack(reconstruction_losses).mean()
    cmae_loss.backward()
    optimizer.step()

    cmae.eval()
    with torch.no_grad():
        fused = cmae(features, global_ft=True)
    top_k = min(int(parameters.get("hypergraph_k", 12)), fused.shape[0])
    distances = torch.cdist(fused, fused)
    neighbor_indices = distances.topk(top_k, dim=1, largest=False).indices
    incidence = torch.zeros_like(distances)
    columns = torch.arange(fused.shape[0], device=device).unsqueeze(1).expand_as(neighbor_indices)
    incidence[neighbor_indices, columns] = 1
    row_norm = incidence.sum(dim=1, keepdim=True).clamp_min(1).reciprocal()
    column_norm = incidence.sum(dim=0, keepdim=True).clamp_min(1).reciprocal()
    propagation = (row_norm * incidence) @ (column_norm * incidence).T
    n_classes = int(labels.max().item()) + 1
    hgnn = HGNN(n_classes, fused.shape[1]).to(device)
    hgnn.train()
    hgnn_optimizer = torch.optim.SGD(hgnn.parameters(), lr=float(parameters.get("hgnn_learning_rate", 0.001)), momentum=0.9)
    hgnn_optimizer.zero_grad()
    logits, reconstructed_logits, embeddings, reconstructed_embeddings, _ = hgnn(fused.detach(), propagation)
    hgnn_loss = 0.05 * (nn.functional.cross_entropy(logits[train_mask], labels[train_mask]) + nn.functional.cross_entropy(reconstructed_logits[train_mask], labels[train_mask])) + 0.9 * nn.functional.mse_loss(embeddings[train_mask], reconstructed_embeddings[train_mask])
    hgnn_loss.backward()
    hgnn_optimizer.step()

    hgnn.eval()
    started = time.perf_counter()
    with torch.no_grad():
        retrieval_features = hgnn(fused, propagation, global_ft=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    retrieval_np = retrieval_features.detach().cpu().numpy()
    query_indices = np.flatnonzero(query_mask)
    target_indices = np.flatnonzero(target_mask)
    query_count = min(int(parameters.get("query_count", 3)), len(query_indices))
    result_items = []
    for query_index in query_indices[:query_count]:
        cosine = scipy.spatial.distance.cdist(retrieval_np[[query_index]], retrieval_np[target_indices], "cosine")[0]
        order = np.argsort(cosine)[: min(5, len(target_indices))]
        result_items.append({
            "query_index": int(query_index),
            "query_label": int(labels[query_index].item()),
            "neighbors": [
                {"target_index": int(target_indices[i]), "target_label": int(labels[target_indices[i]].item()), "cosine_distance": float(cosine[i])}
                for i in order
            ],
        })

    run_dir = Path(request["output_dir"]) / request["run_id"]
    feature_path = run_dir / "hgm2r_smoke_embeddings.npy"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, retrieval_np)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "scipy": scipy.__version__, "device": str(device)}
    if device.type == "cuda":
        environment.update({"gpu": torch.cuda.get_device_name(device), "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1048576})
    _write(response_path, {
        "status": "succeeded",
        "predictions": {"items": result_items, "embedding_path": str(feature_path.resolve())},
        "artifacts": [str(feature_path.resolve())],
        "metrics": {"num_samples": int(fused.shape[0]), "modalities": 3, "embedding_dim": int(retrieval_np.shape[1]), "cmae_one_step_loss": float(cmae_loss.item()), "hgnn_one_step_loss": float(hgnn_loss.item())},
        "timing": {"retrieval_forward_seconds": elapsed},
        "environment": environment,
        "warnings": ["Upstream publishes no HGM2R checkpoint. This run uses bundled ESB features and one optimization step to validate the complete CMAE-hypergraph-HGNN-retrieval path; its ranking is not a paper-result reproduction."],
        "message": "HGM2R real-feature integration smoke completed.",
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
