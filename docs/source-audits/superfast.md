# SuperFast source audit

Status: `adapted` (checkpoint-backed synthetic-sample integration)

- Source: https://github.com/lisiqi19971013/SuperFast
- Pinned commit: `b4d29efc75c4a6ca07de4c5f56f555de48b65741`
- Main-code license: MIT
- Required `slayerPytorch` dependency: GPL-3.0 and compiled C++/CUDA code
- Task: event-based video frame interpolation
- Upstream inference scripts: `test_THU_HSEVI.py` and `test_HSERGB.py`
- Declared environment: PyTorch 1.9.0, torchvision 0.10.0, CUDA 11.1,
  `slayerPytorch`
- Checkpoint: official `ckpt_THU_HSEVI.pth`, `418,931,023` bytes, SHA256
  `4828b7647b77198795165d2a3c18c37c2773b4c785fd06a96da9df296965ce88`;
  stored outside Git
- Known static issues: hard-coded GPU IDs and paths, unconditional `.cuda()`
  calls, `DataParallel`, and `torch.load()` without `map_location`
- Local source location: ignored checkout at `third_party/SuperFast`
- HyperComp entry: `hypercomp.plugins.superfast.adapter:run_superfast`

The worker exposes one THU-HSEVI scene and sample index through the common
request/result boundary. It keeps the upstream model code unchanged and
normalizes checkpoint, device, input, and output paths around it.

On 2026-08-11 the pinned source, compiled slayerPytorch CUDA extension, official
checkpoint, upstream data loader/model API, and HyperComp unified entry were
run on an RTX 3090 Linux server. Both inference paths produced a 64 x 64
interpolated frame from the same deterministic synthetic scene. This proves
the adapter boundary and advances the component to `adapted`.

The input is structurally compatible synthetic data, not an official
THU-HSEVI sample. Therefore this run is not a paper reproduction and does not
advance the separate official-data gates `upstream_verified` or
`platform_verified`. See
`docs/validation/superfast-remote-adaptation-2026-08-11.md`.
