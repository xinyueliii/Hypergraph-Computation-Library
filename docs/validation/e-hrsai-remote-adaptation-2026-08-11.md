# E-HRSAI remote adaptation record

Date: 2026-08-11

Status reached: `adapted` using the official checkpoint and a synthetic tensor
layout. This is not an official-dataset or paper-metric reproduction.

## Fixed source and asset

- Source commit: `8cf3a1c0d1befb57d0d95df4c4a6c90f6c500ceb`
- Checkpoint: `checkpoint_main.pth`
- Size: `378071377` bytes
- SHA256: `6ef548d02f249b70f904093075b38e5ec427f6b23d085f8e9c25750564e06fe9`
- External path: `/root/autodl-tmp/hypercomp-data/checkpoints/e-hrsai/checkpoint_main.pth`

All 440 checkpoint state entries matched the constructed model.

## Runtime

- Server: AutoDL Ubuntu 20.04, NVIDIA GeForce RTX 3090 24 GB
- Environment: `/root/autodl-tmp/hypercomp-data/envs/event-cu113`
- Python: 3.8.10
- PyTorch: 1.10.0+cu113
- CUDA toolkit: 11.3
- GCC: 9
- Environment specification: `environments/event-remote-cu113.yml`

This is a tested compatibility environment. It does not reproduce the upstream
PyTorch 1.9/CUDA 11.1 declaration byte for byte.

## Unified-entry run

```bash
HYPERCOMP_E_HRSAI_PYTHON=/root/autodl-tmp/hypercomp-data/envs/event-cu113/bin/python \
python -m hypercomp run \
  --task event_frame_synthetic_aperture_imaging \
  --model e-hrsai --action smoke --device cuda:0 \
  --checkpoint /root/autodl-tmp/hypercomp-data/checkpoints/e-hrsai/checkpoint_main.pth
```

The worker uses the tensor shapes documented in the upstream model source:
frames `[1,3,51,256,256]`, masks `[1,3,17,256,256]`, event voxels
`[1,3,36,256,256]`, and two event-flow tensors `[1,5,256,256]`.

- Result: `succeeded`
- Output shape: `[1,3,1024,1024]`
- Worker inference: `0.6100862696766853` seconds
- Platform total: `7.361635655164719` seconds
- Peak allocated GPU memory: `7307.3935546875` MiB
- Artifact: `/root/autodl-tmp/hypercomp-data/runs/e-hrsai-unified-20260811/synthetic_reconstruction.png`

Request, worker request/response, stdout, stderr, output image, and final result
are stored in the same external run directory and are excluded from Git.

## Remaining gate

The synthetic zero tensors prove model loading, CUDA execution, output
serialization, environment isolation, and unified routing. They do not prove
the event/frame preprocessing path or task accuracy. Official dataset input,
metric calculation, repeatability, failure cases, and a clean rebuild remain
required before `platform_verified`.
