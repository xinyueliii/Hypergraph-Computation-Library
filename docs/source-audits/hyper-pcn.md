# Hyper-PCN source audit

Status: `adapted` with an official-PCN-checkpoint synthetic smoke

- Source: https://github.com/Rinfly/Hyper-PCN
- Pinned commit: `e890be8653af8ac55ee73d573936d50d461e80e5`
- License: not declared in the pinned repository; upstream source remains an
  ignored runtime checkout and is not redistributed in HyperComp Git history
- Task: hypergraph-based point-cloud completion
- Upstream entry: `main.py --test --ckpts WEIGHT --config CONFIG`
- Declared stack: Ubuntu 22.04/24.04, Python 3.9, PyTorch 2.1.1,
  torchvision 0.16.1, CUDA 11.8, GCC 9
- Input: PCN/ShapeNet partial point clouds described by the selected config
- Output: dense completed point clouds and completion metrics
- Checkpoint: external Tsinghua Cloud directory; size, mapping, license, and hash pending
- Risks: compiled Chamfer Distance and PointNet2 CUDA extensions; the pinned
  setup files hard-code obsolete architectures and are patched reproducibly to
  `TORCH_CUDA_ARCH_LIST=8.6` by `scripts/prepare_hyper_pcn_source.py`
- HyperComp mode: worker process in shared `visual-cu121`, with the shared CUDA
  12.1 compiler-only toolchain for extensions

Because licensing is undeclared, only HyperComp-owned manifests and adapters
are versioned. Runtime source is fetched by pinned commit on the server.

On the remote RTX 3090, the shared `visual-cu121` environment compiled both
extensions using a small shared CUDA 12.1 toolchain. All 465 PCN checkpoint
entries matched, and the unified smoke produced 1024-, 2048-, and 16384-point
stages. This validates integration, not PCN benchmark accuracy. See
`docs/validation/hyper-pcn-remote-adaptation-2026-08-11.md`.
