# Count Anything remote adaptation record

Status: `adapted`

- Platform: AutoDL, Ubuntu 20.04, NVIDIA RTX 3090 24 GB
- Runtime: Python 3.11.15, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126
- Source commit: `0fffaefb3bbdcd930c135dcc02e2e359a038f2cb`
- Checkpoint: `count-anything.pt`, 3,565,313,633 bytes, SHA256 `2f84be3c3af5508388b732827e725cb6d6c4dd826ffda7f80de234ebd1ef0f92`
- Input: YOLOv13 bundled `bus.jpg`; query `person`
- Result: count 4 with four scored point predictions and a rendered image
- Run: `count-anything-20260812-121714`
- Timing: 24.78 seconds worker inference; 26.83 seconds platform total

The first attempt reused `visual-cu121`, as required by the shared-environment
policy. It reproduced `ModuleNotFoundError: torch.nn.attention` under PyTorch
2.2.2. Only then was the component-specific runtime retained. `pip check` in
that runtime reports `decord 0.6.0 is not supported on this platform`; actual
Linux import and end-to-end single-image inference both succeed, so this is
recorded as upstream package metadata rather than a runtime failure. The shared
visual environment remains `pip check` clean.

This is a fixed single-image inference validation, not CLOC benchmark
reproduction or multi-GPU training validation.
