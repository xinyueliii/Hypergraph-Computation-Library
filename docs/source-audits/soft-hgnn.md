# SoftHGNN source audit

Status: superseded by local untrained `adapted` smoke evidence on 2026-08-10

- Source: https://github.com/Mengqi-Lei/SoftHGNN
- Pinned commit: `3f46b20eeb226616a8af89f321b2e1dc3ee3634f`
- License: MIT
- Integrated scope: basic SoftHGNN module plus CIFAR-10/CIFAR-100 image
  classification; crowd counting and object detection are not exposed yet
- Upstream model: `ImageClassification/models/vit_softhgnn.py`
- Upstream training entry: `ImageClassification/train.py`
- Basic reusable modules: `SoftHGNN_BasicModule/SoftHGNN.py` and
  `SoftHGNN_BasicModule/SoftHGNN_SeS.py`
- Declared image-classification environment: PyTorch 2.4.1+cu118,
  torchvision 0.19.1+cu118, einops 0.8.1, and wandb 0.19.11
- Data: CIFAR-10 and CIFAR-100 are downloaded by torchvision in the upstream
  training script
- Checkpoint: no pretrained SoftHGNN checkpoint was found in the repository or
  linked from its README
- Local source location: ignored checkout at `third_party/SoftHGNN`
- HyperComp entry: `hypercomp.plugins.soft_hgnn.adapter:run_soft_hgnn`

The upstream classification repository provides training but no standalone
prediction script. Its resume path expects a `net` key and a filename that do
not match the `model` key and patch-qualified filename written by the same
training script. The HyperComp worker accepts `model`, `net`, `state_dict`, or
`model_state` mappings and converts image predictions into top-k class results.
The model and common worker have now completed a deterministic GPU smoke in the
shared Windows runtime. No trained checkpoint was loaded, so this validates
integration only and not classification quality. See
`docs/validation/soft-hgnn-local-smoke-2026-08-10.md`.
