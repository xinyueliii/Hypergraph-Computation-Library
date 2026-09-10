# PVRNet remote adaptation record

Status: `adapted`

- Platform: AutoDL, Ubuntu 20.04, NVIDIA RTX 3090 24 GB
- Shared runtime: Python 3.11.15, PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121
- Source commit: `7b07d62788e67b4052c36b9ca37f6163234a4648`
- Official final checkpoint: 607,295,164 bytes, SHA256 `8bd231f3c28da9304c55584725f6897efda1d8bb16af03ee9c2441ccd763f235`
- Fixed sample: Manifold40 `piano_0303.obj`, converted deterministically to 1,024 points and 12 rendered views
- Run: `pvrnet-20260812-123906`
- Result: `piano`, probability 0.921859; retrieval feature exported
- Timing: 0.31 seconds worker inference; 3.81 seconds platform total; 461.53 MiB peak allocated GPU memory

The compatibility layer disables the obsolete automatic AlexNet ImageNet
download, strips optional `module.` checkpoint prefixes, parameterizes the
device, and duplicates the one-item batch internally because the pinned model
uses an unsafe `squeeze()` operation. Only the first prediction is returned.
The source architecture and official final PVRNet weights are retained.

The fixed sample validates the HyperComp inference and feature-extraction
path. It is not a full ModelNet40 accuracy or retrieval-mAP reproduction.
