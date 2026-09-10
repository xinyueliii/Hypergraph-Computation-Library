# Hyper-YOLO source audit

Status: superseded by local `adapted` evidence on 2026-08-10

- Source: https://github.com/iMoonLab/Hyper-YOLO
- Pinned release: `v0.1`
- Pinned commit: `9bfdabd8b97b5ee5da04e5df30d140a9d15557c5`
- License: AGPL-3.0-only
- Task: object detection; upstream also contains instance segmentation paths
- Core upstream API: `from ultralytics import YOLO`
- Upstream prediction script: `ultralytics/models/yolo/detect/predict.py`
- Hypergraph modules: `MessageAgg`, `HyPConv`, and `HyperComputeModule` in
  `ultralytics/nn/modules/block.py`
- Declared environment: Python 3.8.16, PyTorch 2.0.1, torchvision 0.15.2
- Checkpoints: ten assets are published in the upstream `v0.1` release; no
  asset has been downloaded or hashed in this stage
- Local source location: ignored checkout at `third_party/Hyper-YOLO`
- HyperComp entry: `hypercomp.plugins.hyper_yolo.adapter:run_hyper_yolo`

The code-first adapter accepts the common task request, launches a standalone
worker, invokes the upstream YOLO API, and converts boxes and rendered images
to a common result. The original static conclusion has now been superseded:
the official Hyper-YOLO-N checkpoint passed both the upstream API and HyperComp
GPU paths in the shared Windows runtime. See
`docs/validation/hyper-yolo-local-adaptation-2026-08-10.md`.
