# SoftHGNN object-detection remote adaptation record

## Result

`YOLO11-SoftHGNN-N` completed a unified-entry single-image GPU architecture
smoke on the AutoDL RTX 3090. The component is `adapted`, not
`platform_verified` and not a trained-model reproduction.

## Fixed inputs and source

- Source commit: `3f46b20eeb226616a8af89f321b2e1dc3ee3634f`
- Model config: `ObjectDetection/ultralytics/cfg/models/11-SoftHGNN/yolo11-SoftHGNN.yaml`
- Input: upstream Ultralytics `assets/bus.jpg`
- Checkpoint: none published; deterministic random initialization with seed 2026
- Run ID: `soft-hgnn-detection-20260812-01`

## Environment and result

- Environment: shared `visual-remote-cu121`
- Python: 3.11.15
- PyTorch: 2.2.2+cu121
- Ultralytics: 8.3.63 from the pinned SoftHGNN source tree
- GPU: NVIDIA GeForce RTX 3090
- Image size: 640
- Worker inference: 1.4211 seconds
- Platform total: 4.6632 seconds
- Peak allocated GPU memory: 204.58 MiB
- Rendered artifact: `/root/autodl-tmp/hypercomp-data/runs/soft-hgnn-detection-20260812-01/rendered/0000-bus.jpg`

The run returned no detections, which is expected for an untrained detector.
It proves model parsing, custom SoftHGNN modules, image preprocessing, CUDA
forward execution, postprocessing, artifact creation, and the unified result
contract. It does not prove detection quality. A meaningful `predict` action
continues to require a trained matching checkpoint.
