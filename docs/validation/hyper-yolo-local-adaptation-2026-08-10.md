# Hyper-YOLO 本地适配记录（2026-08-10）

## 结论

Hyper-YOLO-N 已在本机 RTX 3060 Laptop GPU 上完成原项目 API 与 HyperComp 统一入口的单图推理，组件状态由 `source_audited` 更新为 `adapted`。这不是 COCO 精度复现或平台最终验收。

## 固定来源与资产

- 源码：iMoonLab/Hyper-YOLO，release `v0.1`，提交 `9bfdabd8b97b5ee5da04e5df30d140a9d15557c5`。
- checkpoint：官方 release 资产 `hyper-yolon.pt`。
- 外部位置：`D:\HyperCompAssets\checkpoints\hyper-yolo\hyper-yolon.pt`。
- 大小：8,567,703 bytes。
- SHA256：`C51808F9B019C136ECD47AB50E59AC3E93CCF41DE2F54E11B70542A217BC8932`。
- 固定输入：YOLOv13 上游源码附带的 `ultralytics/assets/bus.jpg`。

## 环境与本机预评估

- 环境 ID：`shared-win-cu121`。
- Python：3.11.15。
- PyTorch：2.2.2+cu121。
- Torchvision：0.17.2+cu121。
- GPU：RTX 3060 Laptop GPU，6 GB。
- 结论：`local_safe`。Hyper-YOLO 核心模块能够在共享环境导入，官方 N 模型权重仅约 8.6 MB。

## 原项目路径结果

通过 Hyper-YOLO 源码目录中的 `ultralytics.YOLO` API，以 batch 1、640 分辨率和 CUDA 设备运行固定图片：

- 检测数：6。
- 类别：4 个 person、1 个 bus、1 个 skateboard。
- 上游日志推理时间：658.3 ms。
- 完整调用耗时：约 5.91 s。
- PyTorch 峰值分配显存：约 185.5 MB。
- 推理前后没有新增常驻显存；温度 52→53 摄氏度。

## HyperComp 统一入口结果

- run ID：`hyper-yolo-platform-20260810-01`。
- 状态：`succeeded`。
- worker 时间：约 2.02 s。
- 平台总时间：约 6.52 s。
- 结果包含统一检测框、类别、置信度、日志和渲染图片。
- 渲染图片 SHA256：`CA8D591DDE069D48241B70301313F4296726D59251455AA3CCB6D07BA2DCD435`。

YOLOv13 与 Hyper-YOLO 都自带名为 `ultralytics` 的源码。两者共享同一个 Python 环境，但由 HyperComp 在独立工作进程中切换源码根目录，避免同一进程中的包名冲突。
