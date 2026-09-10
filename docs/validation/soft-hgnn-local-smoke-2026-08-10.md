# SoftHGNN 本地 smoke 记录（2026-08-10）

## 结论

SoftHGNN 图像分类模型已在共享环境完成 GPU 前向，并通过 HyperComp 统一入口完成未训练权重的确定性 smoke。组件状态更新为 `adapted`，但没有公开 checkpoint，因此不能声称已经获得有效分类能力或复现论文精度。

## 环境与本机预评估

- 环境 ID：`shared-win-cu121`。
- Python：3.11.15。
- PyTorch：2.2.2+cu121。
- Torchvision：0.17.2+cu121。
- 新增兼容依赖：einops 0.8.1。为共享运行时补充 scikit-learn 1.6.1 后，HGNN/HGNN+ 回归测试也通过；`pip check` 无损坏依赖。
- GPU：RTX 3060 Laptop GPU，6 GB。
- 结论：`local_safe`，适合 CIFAR 分辨率的 batch 1 推理和小批量实验；大批量正式训练需另外评估。

## 模型前向结果

使用上游 `ViTWithSoftHGNN` 默认 CIFAR-10 结构：

- 参数量：18,815,473。
- 输入：`(1, 3, 32, 32)`。
- 输出：`(1, 10)`。
- 单次 GPU 前向：约 0.242 s。
- PyTorch 峰值分配显存：约 90.3 MB。

## HyperComp 统一入口结果

- run ID：`soft-hgnn-platform-20260810-01`。
- 状态：`succeeded`。
- worker 推理时间：约 0.382 s。
- 平台总时间：约 4.22 s。
- checkpoint：未加载。
- 权重：根据请求 seed 确定性随机初始化。

`smoke` 动作允许无 checkpoint，用于检查模型构建、图像预处理、GPU 前向和统一结果结构；返回结果带有明确警告。`predict` 动作仍强制要求训练 checkpoint，以免把随机分数误当成有效预测。

另行执行了缺少 checkpoint 的 `predict` 负向检查，平台按预期返回 `failed` 和非零退出码，没有把无权重运行误报为预测成功。
