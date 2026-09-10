# YOLOv13 source audit

Current status: `adapted`

- Source: https://github.com/iMoonLab/yolov13
- Pinned commit: `73289949533efac82bb5f72ec19b746618656bd2`
- License: AGPL-3.0-only
- Task: real-time object detection
- Upstream API: `from ultralytics import YOLO`
- Declared environment: Python 3.11, PyTorch 2.2.2, torchvision 0.17.2
- Optional acceleration: a Linux FlashAttention 2.7.3 wheel; source inspection
  shows a PyTorch scaled-dot-product-attention fallback
- Checkpoints: upstream release publishes N, S, L, and X variants; the N
  checkpoint has been downloaded outside Git and hashed
- Local source location: ignored checkout at `third_party/YOLOv13`
- HyperComp entry: `hypercomp.plugins.yolov13.adapter:run_yolov13`

YOLOv13 and Hyper-YOLO now use the same HyperComp Ultralytics-family worker,
while retaining separate source checkouts, manifests, Python executables, and
license records. The worker accepts images or videos and normalizes boxes,
class labels, confidences, rendered images, logs, and environment metadata.
The original inference path and the HyperComp unified entry both produced a
valid result on 2026-08-10. See
`docs/validation/yolov13-local-adaptation-2026-08-10.md` for the exact
environment, asset hashes, timings, resource observations, artifacts, and
remaining verification limits.
