# Hyper-PCN remote adaptation record

Date: 2026-08-11

Status reached: `adapted` with the official PCN checkpoint and deterministic
synthetic partial points. This is not a PCN dataset or paper-metric reproduction.

## Source and checkpoint

- Source commit: `e890be8653af8ac55ee73d573936d50d461e80e5`
- Source license: not declared; source stays in ignored runtime checkout
- Checkpoint: `pcn.pth`
- Size: `264775090` bytes
- SHA256: `ad7c22bc5e42dfbfd227cc601628e69346b2e11b5a93b0d9af6f67c5e976ded9`
- External path: `/root/autodl-tmp/hypercomp-data/checkpoints/hyper-pcn/pcn.pth`

All 465 checkpoint entries matched the constructed model.

## Shared environment and compiler

- Server: AutoDL Ubuntu 20.04, NVIDIA GeForce RTX 3090 24 GB
- Reused Python environment: `visual-cu121`
- Python: 3.11.15
- PyTorch: 2.2.2+cu121
- Shared compiler-only toolchain: CUDA 12.1.105, approximately 184 MB
- Source compatibility patch: obsolete architecture list replaced by an
  environment-overridable default; build fixed to RTX 3090 architecture 8.6
- Compiled packages: Chamfer 2.0.0 and PointNet2 Ops 3.0.0

No separate Hyper-PCN Python/PyTorch environment was retained. The initial
partial venv attempt was stopped and removed after the shared-environment rule
was clarified. Rebuild commands are in `scripts/build_hyper_pcn_extensions.sh`.

## Unified-entry result

```bash
HYPERCOMP_HYPER_PCN_PYTHON=/root/autodl-tmp/hypercomp-data/envs/visual-cu121/bin/python \
python -m hypercomp run \
  --task point_cloud_completion --model hyper-pcn --action smoke \
  --device cuda:0 \
  --checkpoint /root/autodl-tmp/hypercomp-data/checkpoints/hyper-pcn/pcn.pth \
  --parameter point_count=2048
```

- Result: `succeeded`
- Stage shapes: `[1,1024,3]`, `[1,2048,3]`, `[1,16384,3]`
- Worker inference: `0.44572270661592484` seconds
- Platform total: `3.6636564880609512` seconds
- Peak allocated GPU memory: `299.736328125` MiB
- Artifact: `/root/autodl-tmp/hypercomp-data/runs/hyper-pcn-unified-20260811/completed_point_cloud.npy`

The run directory also contains request, worker request/response, stdout,
stderr, and final result JSON files.

## Remaining gate

The random 2048-point input validates checkpoint loading, compiled CUDA ops,
model execution, unified routing, and output serialization. A legal fixed PCN
sample, dataset preprocessing, Chamfer metrics, repeatability, failure cases,
and clean rebuild are still required before `platform_verified`.
