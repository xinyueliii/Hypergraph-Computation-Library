# PVRNet source audit

Status: `adapted` (runtime evidence: `docs/validation/pvrnet-remote-adaptation-2026-08-12.md`)

- Source: https://github.com/iMoonLab/PVRNet
- Pinned commit: `7b07d62788e67b4052c36b9ca37f6163234a4648`
- License: MIT for code
- Paper: PVRNet: Point-View Relation Neural Network for 3D Shape Recognition
- Task: ModelNet40 point-cloud plus 12-view classification and retrieval-feature extraction
- Upstream entry: `val_pvrnet.py`
- Input: one 1024-point NumPy array and 12 RGB renderings of the same object
- Output: 40-class logits, a 256-dimensional retrieval representation, and
  upstream dataset-level accuracy/mAP when a full labeled split is used
- Checkpoint: official final PVRNet file, 607,295,164 bytes, SHA256
  `8bd231f3c28da9304c55584725f6897efda1d8bb16af03ee9c2441ccd763f235`
- Declared stack: Ubuntu 16.04, Python 3.6, PyTorch 0.4.1, CUDA 9.0
- HyperComp mode: inference-only isolated worker with a small modern compatibility
  layer; first assigned to `visual-remote-cu121`

The source uses only standard PyTorch/torchvision operators for PVRNet itself;
the main migration risks are deprecated torchvision pretrained-model arguments,
hard-coded multi-GPU settings, old NumPy aliases, checkpoint key prefixes, and
dataset path assumptions. HyperComp does not reconstruct CUDA 9 on the RTX 3090.
Instead, it disables unnecessary ImageNet downloads, loads the final checkpoint
with `map_location`, strips `module.` prefixes when present, parameterizes the
device, and accepts one paired point/view sample. The official checkpoint
correctly classified the fixed mesh-derived sample as `piano`, so the component
is now `adapted`; full ModelNet40 accuracy/mAP remains outside this validation.
