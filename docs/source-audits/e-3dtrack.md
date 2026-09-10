# E-3DTrack source audit

Status: `adapted`

- Source: https://github.com/lisiqi19971013/E-3DTrack
- Pinned commit: `9f4d61a0912a6337b7595e87337dd3b144e62a00`
- License: MIT for code; dataset/checkpoint redistribution terms are not stated
- Task: event-based stereo 3D feature tracking
- Upstream entry: `eval.py`; checkpoint, data, output, and GPU paths are hard coded
- Declared stack: Python 3.8, PyTorch 1.9, CUDA 11.1, Kornia 0.6.7,
  einops 0.4.1, OpenCV 4.6
- Input: dataset root with `test.txt` and sequence directories containing stereo
  event tensors, calibration, reference images, and tracks
- Output: `pos_3d_pred.npy` and `pos_3d_gt.npy` per sequence
- Checkpoint: official `ckpt.pth`, 138,960,387 bytes, SHA256
  `862ff7db5e31beb93716c325b91c0120c2f61825ebfbaef28f951f053fb29cb2`;
  downloaded on Windows and transferred to the AutoDL data disk because the
  AutoDL Google Drive connection timed out
- Risks: unconditional `.cuda()` calls and fixed GPU index in upstream evaluation
- HyperComp mode: worker process with parameterized paths; assigned to shared
  `event-cu113`; remote RTX 3090 only

The worker retains the upstream network and replaces only hard-coded execution
values. On 2026-08-12, the checkpoint loaded into `TrackerNetEval` in shared
`event-cu113`, and all state-dict keys matched successfully (34,700,910 model
parameters). The official `E-3DTrack.zip` archive was downloaded, verified as
3,524,467,933 bytes with SHA256
`e729e69c6e868edf1f738a2b5302de9054488d692f41cfd48668fb6d5bdbd4e0`, and
the first official `test.txt` sequence (`airplane_1/20`) was extracted as a
minimal legal input. HyperComp's unified entry completed successfully and
created finite `pos_3d_pred.npy` and `pos_3d_gt.npy` arrays, both shaped
`(167, 7, 3)`. This reaches `adapted`; it is not a full-dataset benchmark or
`platform_verified` result.
