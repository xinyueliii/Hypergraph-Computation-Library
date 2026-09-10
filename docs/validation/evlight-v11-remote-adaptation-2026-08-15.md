# EvLight V11 remote adaptation record

## Result

EvLight V11 completed checkpoint-backed single-sample inference through the
HyperComp unified CLI on the AutoDL RTX 3090. The component is `adapted`, not
`platform_verified`, and the reported sample metrics are not a reproduction of
the CVPR 2024 paper protocol.

## Source, checkpoint, and input

- Source: internal EvLight V11 research snapshot based on
  `https://github.com/EthanLiang99/EvLight`
- Model variant: `egretinex-dual-freq-V11.py`, installed as the active
  `egretinex.py`
- Run ID: `evlight-v11-epoch19-final-sdsd-in-single`
- Input: SDSD-in sequence `pair23`, frame `0003`, resized to `256 x 256`
- Checkpoint: `evlight_v11_epoch19_final_compact_inference.pth`
- Checkpoint SHA256:
  `5d44526e4310a01ff3b0a840a9c541fdef6217738ff5d3883596ed4466dec04b`
- Checkpoint provenance: epoch-19 best model from a completed 20-epoch,
  16-train-sequence, one-validation-sequence fast SDSD-in run. Final fast
  validation was SSIM 0.920420, PSNR 29.9854 dB, and PSNR* 31.4227 dB.
  Optimizer state and 302 separately loaded CLIP parameters were removed from
  the compact inference copy.
- CLIP parameters: OpenAI ViT-B/32, passed as an external runtime asset rather
  than committed to Git.

## Exercised path

The `event_guided_low_light_enhancement` task selected `evlight-v11`, loaded the
isolated upstream source and compact checkpoint, constructed one SDSD-style
frame/event sample, ran CUDA inference, and wrote the low-light input,
prediction, ground truth, illumination map, and blurred input through the
common result contract.

## Environment and result

- Environment: shared `visual-remote-cu121`; no component-specific environment
  was created
- Python: 3.11.15
- PyTorch: 2.2.2+cu121
- Torchvision: 0.17.2+cu121
- CUDA reported by PyTorch: 12.1
- GPU: NVIDIA GeForce RTX 3090
- Worker inference: 0.5884 seconds
- Platform total: 6.7570 seconds
- Peak allocated GPU memory: 594.91 MiB
- Single-sample MSE: 0.00119124
- Single-sample PSNR: 29.2400 dB
- Single-sample SSIM: 0.903300
- `pip check`: no broken requirements after adding `absl-py`, `ftfy`, and
  `regex` to the shared runtime

The prediction was visually inspected against the input and ground truth. It
restored the room structure and object detail without obvious collapse or
severe artifacts, although it was mildly brighter and warmer than the ground
truth. These observations and metrics only validate this fixed quick-subset
checkpoint and sample.

For the same HyperComp sample, the earlier epoch-4 compact checkpoint produced
PSNR 27.4674 dB, SSIM 0.904366, and MSE 0.00179169. The final checkpoint was
selected because it materially improved PSNR and MSE and completed the planned
training, while SSIM changed slightly downward by about 0.0011.

## Asset and replacement boundary

Source snapshots live under ignored `third_party/`; datasets, checkpoints,
CLIP weights, and run outputs live under `/root/autodl-tmp/hypercomp-data/` and
are not tracked by Git. A later checkpoint can replace the current asset
without changing the task, adapter, or worker interface, provided its model
architecture remains compatible.
