# YOLOv13 local adaptation validation

Date: 2026-08-10

Status reached: `adapted`

This record covers one upstream GPU inference and one HyperComp unified-entry
GPU inference. It is not a COCO benchmark reproduction and does not yet meet
the repeated-run gate for `platform_verified`.

## Fixed inputs

- Source repository: `https://github.com/iMoonLab/yolov13`
- Source commit: `73289949533efac82bb5f72ec19b746618656bd2`
- Model: official YOLOv13-N release checkpoint
- Checkpoint URL:
  `https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt`
- Checkpoint size: `10,570,688` bytes
- Checkpoint SHA256:
  `6653035017B0F111F80EC11ED914874EA85699B104AEAC1E46E517D16889D6B7`
- Sample: upstream `ultralytics/assets/bus.jpg`
- Sample size: `137,419` bytes
- Sample SHA256:
  `C02019C4979C191EB739DDD944445EF408DAD5679ACAB6FD520EF9D434BFBC63`

The checkpoint and runtime outputs are stored outside Git under
`D:\HyperCompAssets` and `D:\HyperCompArtifacts`.

## Environment

- Host: Windows, NVIDIA driver `566.03`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM
- Python: `3.11.15`
- PyTorch: `2.2.2+cu121`
- torchvision: `0.17.2+cu121`
- NumPy: `1.26.4`
- OpenCV: `4.10.0`
- Ultralytics/YOLOv13 package: `8.3.63`, editable pinned source
- Environment prefix: `D:\Anaconda\envs\hypercomp-yolov13`
- Reproducible specification: `environments/yolov13-win-cu121.yml`
- Full installed-package snapshot:
  `D:\HyperCompArtifacts\yolov13\environment-freeze.txt`, SHA256
  `353D71F5BCB22C330CFBE89720B24FC2E81CEDB00EF1D7C68C8F89C2DABA5731`

The upstream requirements file references a Linux FlashAttention wheel. It
was not installed on Windows. Source code selected PyTorch
`scaled_dot_product_attention` as its built-in fallback.

During environment reconstruction, current unconstrained packages selected
NumPy 2.x and OpenCV 5.x, which are incompatible with the NumPy ABI expected
by PyTorch 2.2.2. The validated environment pins NumPy 1.26.4 and OpenCV
4.10.0.84. The source also imported `huggingface_hub` without declaring it in
the inspected project dependencies, so the validated specification records it
explicitly. `pip check` reported no broken requirements after these fixes.

## Upstream inference

- Input size: `640`
- Device: CUDA GPU `0`
- Confidence threshold: `0.25`
- Result: success
- Detections: five persons and one bus
- Reported timing: `8.3 ms` preprocessing, `187.0 ms` inference, and
  `178.1 ms` postprocessing
- Output directory: `D:\HyperCompArtifacts\yolov13\upstream-run`

GPU telemetry immediately before and after the short process was:

| Point | Temperature | Power | Used VRAM | GPU utilization |
| --- | ---: | ---: | ---: | ---: |
| Before | 48 C | 8.38 W | 144 MiB | 0% |
| After | 50 C | 16.16 W | 144 MiB | 0% |

This demonstrates that the one-image Nano inference released its GPU memory
after exit and had little persistent impact on the laptop.

## HyperComp unified-entry inference

- Run ID: `yolov13-platform-20260810-01`
- Task/model: `object_detection` / `yolov13`
- Result: `succeeded`
- Detections: six, matching five persons and one bus
- Worker time: `2.2697566999704577` seconds
- Platform total time: `5.660114900034387` seconds
- Worker inference line: `243.5 ms`
- stderr: empty
- Output directory:
  `D:\HyperCompArtifacts\yolov13\hypercomp\yolov13-platform-20260810-01`

The unified entry was selected with the isolated worker interpreter and
writable application-cache locations:

```powershell
$env:HYPERCOMP_YOLOV13_PYTHON = 'D:\Anaconda\envs\hypercomp-yolov13\python.exe'
$env:YOLO_CONFIG_DIR = 'D:\HyperCompCache\yolo-config'
$env:MPLCONFIGDIR = 'D:\HyperCompCache\matplotlib'
python -m hypercomp run --task object_detection --model yolov13 `
  --action predict --run-id yolov13-platform-20260810-01 `
  --input third_party/YOLOv13/ultralytics/assets/bus.jpg `
  --checkpoint D:\HyperCompAssets\checkpoints\yolov13\yolov13n.pt `
  --device cuda:0 --output-dir D:\HyperCompArtifacts\yolov13\hypercomp
```

The run produced the common request, worker request/response, stdout/stderr,
rendered detection image, and final result JSON. The rendered image SHA256 is
`D8833DA09B67A41A1142C9EE8D82793FA8DE8FBC9DD5D9DBCEB2C195C35059D1`.

## Status decision and remaining work

The pinned upstream code produced a valid output and its existing HyperComp
adapter returned a successful common result with preserved artifacts. The
component therefore advances to `adapted`.

It remains below `platform_verified` because the fixed acceptance run has not
yet been repeated, a clean rebuild from the committed environment file has not
been exercised, and failure-path acceptance has not been completed.
