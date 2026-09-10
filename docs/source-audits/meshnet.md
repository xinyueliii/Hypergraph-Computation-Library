# MeshNet source audit

Status: `adapted` with checkpoint-backed single-mesh inference

- Source: https://github.com/iMoonLab/MeshNet
- Pinned commit: `70f9115a121cef71f62d774088771337c3beaf4b`
- License: MIT
- Task: mesh shape classification and feature retrieval
- Upstream entry: `test.py` with `config/test_config.yaml`
- Declared stack: Python 3.8, PyTorch 1.8, CUDA 11.1, PyMeshLab 2021.10,
  SciPy, PyYAML
- Input: processed `.npz` mesh with `faces` and `neighbors`, or supported raw mesh
- Output: class logits/prediction and normalized 256-dimensional retrieval feature
- Original README checkpoint and processed-data links return `Link does not
  exist`. The maintainer supplied a replacement checkpoint in official issue
  27: 17,094,421 bytes, SHA256
  `cff4252d94057d44c33cbab7e549756f7ace1330309ca17b19a55a62b56f4613`.
- A community reply in official issue 26 points to the still-available
  SubdivNet Manifold40 archive: 132,455,392 bytes, SHA256
  `fba848d46a6650c6b182d47a81dd200167e98986203efff2c658a8fa54eea952`.
- HyperComp mode: worker process assigned to shared `event-cu113`; first
  acceptance uses one processed `.npz`

On 2026-08-12, shared `event-cu113` was extended with PyMeshLab 2021.10 and
Rich 13.9.4, then `pip check` passed. The upstream preprocessor converted
`piano_0303.obj` to an official-format NPZ. The HyperComp unified entry loaded
the replacement checkpoint on RTX 3090, predicted class index 6, and saved a
256-dimensional retrieval feature. Inference took 0.0402 seconds inside the
worker and 5.689 seconds end to end. This is a one-sample adapter validation,
not full ModelNet40 accuracy or retrieval reproduction.
