# HGNN Family Local Adaptation Evidence

Date: 2026-08-09

## Scope

This record validates the HyperComp request/result boundary for the native
`HGNN` and `HGNNP` (`HGNN+`) implementations already present in DHG v0.9.7.
It does not claim reproduction of the papers' benchmark accuracy.

The fixed smoke sample is a small, deterministic synthetic hypergraph with 12
vertices, 5 hyperedges, 6 input features, and 3 classes. It is intended only
to exercise construction, message passing, optimization, checkpoint handling,
inference, metrics, and artifact serialization without downloading data.

## Verified environment

- Windows 10 22H2
- Python 3.10.16
- PyTorch 2.7.0+cu126
- CUDA runtime build 12.6
- NVIDIA GeForce RTX 3060 Laptop GPU
- DHG 0.9.7 source in this repository

## Acceptance results

- Component registry discovers both native entry points.
- HGNN and HGNN+ complete CPU training and inference.
- HGNN and HGNN+ complete CUDA training and inference on `cuda:0`.
- Checkpoints are saved and restored through a separate evaluate request.
- A non-default hidden dimension is restored from checkpoint metadata.
- Each successful run writes request, environment, prediction, metric,
  checkpoint, and result artifacts.
- Unknown components return a structured failed result with traceback evidence.

Commands used:

```powershell
python -m unittest discover -s tests -p "test_registry.py" -v
python -m unittest discover -s tests -p "test_native_vertex_classification.py" -v
python -m hypercomp run --component hgnn --action smoke --device cuda:0
python -m hypercomp run --component hgnnp --action smoke --device cuda:0
```

## Status boundary

Both model manifests are advanced to `adapted`: the common platform boundary
works and has local CPU/GPU evidence. `platform_verified` remains pending a
clean environment recreation and a maintained representative dataset/sample
with durable acceptance outputs. Paper-level reproduction remains a separate
future gate. The complete preserved upstream DHG test suite has not yet been
run because the currently available CUDA environment does not contain
`pytest`; the platform tests above use Python's standard `unittest` runner.
