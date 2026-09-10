# E-HRSAI source audit

Status: `adapted` with an official-checkpoint synthetic smoke

- Source: https://github.com/lisiqi19971013/E-HRSAI
- Pinned commit: `8cf3a1c0d1befb57d0d95df4c4a6c90f6c500ceb`
- License: MIT for code; dataset/checkpoint redistribution terms are not stated
- Task: event-assisted high-resolution synthetic-aperture imaging
- Upstream entry: `eval.py --folder DATA --ckpt WEIGHT --opFolder OUTPUT`
- Declared stack: Python 3.8, PyTorch 1.9, CUDA 11.1, OpenCV 4.6
- Input: dataset root with `test.txt`; each scene contains frames, `events.npy`,
  `ts.npy`, `flow.npy`, masks, and ground truth
- Output: reconstructed and ground-truth PNG pairs
- Checkpoint: official external Tsinghua Cloud link; size and SHA256 pending
- Risks: CUDA-only tensor moves, JIT-compiled deformable convolution and fused
  operators, legacy binary stack
- HyperComp mode: worker process in shared `event-cu113`; remote RTX 3090
  acceptance only

The source audit preceded a remote RTX 3090 run. The official checkpoint now
loads with all 440 state entries matched, and both a direct model forward and
the HyperComp unified smoke path produce a 1024 x 1024 reconstruction from the
upstream-documented synthetic tensor layout. This validates the adapter but not
official-dataset accuracy. See `docs/validation/e-hrsai-remote-adaptation-2026-08-11.md`.
