# SoftHGNN object-detection source audit

Status: `adapted`; runtime evidence is recorded in
`docs/validation/soft-hgnn-detection-remote-adaptation-2026-08-12.md`.

- Source: https://github.com/Mengqi-Lei/SoftHGNN
- Pinned commit: `3f46b20eeb226616a8af89f321b2e1dc3ee3634f`
- Paper: Soft Hypergraph Neural Networks for General Visual Recognition
- Paper URL: https://link.springer.com/article/10.1007/s11263-026-02656-2
- License: MIT
- Integrated architecture: `YOLO11-SoftHGNN-N`
- Upstream config: `ObjectDetection/ultralytics/cfg/models/11-SoftHGNN/yolo11-SoftHGNN.yaml`
- Checkpoint: no SoftHGNN detection checkpoint is published in the repository.
- Entry point: `hypercomp.plugins.soft_hgnn_detection.adapter:run_soft_hgnn_detection`

The object-detection subtree is a complete modified Ultralytics distribution
and includes the `F2SoftHG`, `ShapeAlignConv`, and `MergeConv` implementation.
It also contains YOLOv12-SoftHGNN, but the first integration deliberately uses
YOLO11-SoftHGNN because it avoids the YOLOv12 FlashAttention dependency and
matches the existing shared PyTorch 2.2.2/torchvision 0.17.2 runtime.

Without a trained checkpoint, `smoke` builds the official N-scale architecture
with deterministic random initialization, runs image preprocessing and GPU
forward inference, then emits the normal structured detection result. This is
architecture and platform validation only; random detections are not treated
as a meaningful prediction or paper-result reproduction.
