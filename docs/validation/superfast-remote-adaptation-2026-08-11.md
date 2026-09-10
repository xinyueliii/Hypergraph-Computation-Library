# SuperFast 远程适配验证记录

日期：2026-08-11

达到状态：`adapted`

本记录覆盖 slayerPytorch CUDA 扩展验证、官方 checkpoint 校验、上游数据加载/模型 API 推理，以及 HyperComp 统一入口推理。输入为确定性生成的 THU-HSEVI 结构兼容样例，不是官方数据集，因此本记录不构成论文结果复现，也不满足 `upstream_verified` 或 `platform_verified` 的官方样例门槛。

## 固定源码与资产

- SuperFast：`https://github.com/lisiqi19971013/SuperFast`
- SuperFast commit：`b4d29efc75c4a6ca07de4c5f56f555de48b65741`
- slayerPytorch commit：`01beeeb6a181546d6c6830382ce6086bfc587836`
- checkpoint：上游发布的 `ckpt_THU_HSEVI.pth`
- checkpoint 大小：`418,931,023` bytes
- checkpoint SHA256：`4828b7647b77198795165d2a3c18c37c2773b4c785fd06a96da9df296965ce88`
- checkpoint 外部位置：`/root/autodl-tmp/hypercomp-data/checkpoints/superfast/ckpt_THU_HSEVI.pth`
- 合成样例生成器：`scripts/create_superfast_synthetic_sample.py`
- 合成样例外部位置：`/root/autodl-tmp/hypercomp-data/datasets/superfast/synthetic-scene-64`

生成器固定产生 200 张 64 x 64 灰度帧、`ts_frame.txt` 和 40,000 个 `[t, x, y, p]` 事件，并在数据目录写入 `SYNTHETIC_SAMPLE.txt` 防止其被误认为官方样例。数据和 checkpoint 均不进入 Git。

## 服务器与环境

- 平台：AutoDL，Ubuntu 20.04，glibc 2.31
- GPU：NVIDIA GeForce RTX 3090，24 GB
- Python：3.8.10
- PyTorch：1.10.0+cu113
- torchvision：0.11.1+cu113
- CUDA toolkit / nvcc：11.3
- GCC：9
- SuperFast worker：`/root/miniconda3/bin/python`
- HyperComp core：`/root/autodl-tmp/hypercomp-data/envs/core-py310`，Python 3.10.20

本次使用的是经过实测的兼容环境，而不是逐字复刻上游声明的 PyTorch 1.9 / CUDA 11.1。原因是服务器基础镜像已提供 PyTorch 1.10.0+cu113 和完整 CUDA 11.3 工具链，实际编译与推理均通过。对应环境规格记录在 `environments/superfast-remote-cu113.yml`。

HyperComp 主包要求 Python 3.10 或更高；直接在 Python 3.8 中运行 CLI 会在 `dataclass(slots=True)` 处失败。因此保留轻量 Python 3.10 调度环境，并通过 `HYPERCOMP_SUPERFAST_PYTHON` 调用 Python 3.8 SuperFast worker。这是确认过的版本边界，不是推测性隔离。

## slayerPytorch CUDA 验证

slayerPytorch 从固定源码构建成功。随后在 RTX 3090 上直接调用 `slayerCuda.conv`，得到：

- 输出形状：`(1, 1, 1, 1, 16)`
- 输出求和：`5.0`

这证明自定义 CUDA 扩展不仅完成安装，而且能够在当前 GPU 上执行。没有采用 slayerPytorch 的未固定 requirements 覆盖已有 PyTorch 环境。

## 上游 API 推理

使用上游 `UHSEDataset`、`FusionModel`、`DataParallel` 和官方 checkpoint 读取合成场景第 0 个样例并推理：

- 数据张量：双向事件体 `(2, 64, 64, 30)`，图像 `(1, 64, 64)`，事件表面 `(2, 64, 64, 5)`，体素网格 `(5, 64, 64)`
- 输出形状：`[1, 1, 64, 64]`
- 输出最小值：`0.015221902169287205`
- 输出最大值：`0.7381235957145691`
- 输出均值：`0.3682440519332886`
- 模型推理时间：`0.08220026642084122` 秒
- PyTorch 最大分配显存：`909.69580078125` MiB
- 输出：`/root/autodl-tmp/hypercomp-data/runs/superfast-upstream-synthetic/interpolated.png`

该步骤调用上游类和数据加载逻辑，但没有运行官方测试数据及论文指标，所以不标记为 `upstream_verified`。

## HyperComp 统一入口推理

执行命令：

```bash
HYPERCOMP_SUPERFAST_PYTHON=/root/miniconda3/bin/python \
/root/autodl-tmp/hypercomp-data/envs/core-py310/bin/python -m hypercomp run \
  --task event_based_video_frame_interpolation \
  --model superfast \
  --action predict \
  --run-id superfast-synthetic-unified \
  --device cuda:0 \
  --checkpoint /root/autodl-tmp/hypercomp-data/checkpoints/superfast/ckpt_THU_HSEVI.pth \
  --input /root/autodl-tmp/hypercomp-data/datasets/superfast/synthetic-scene-64 \
  --output-dir /root/autodl-tmp/hypercomp-data/runs \
  --parameter sample_index=0 \
  --parameter number_of_time_bins=15
```

结果：

- HyperComp 状态：`succeeded`
- 实际数据源索引：`1`
- 输出分辨率：`[64, 64]`
- worker 推理时间：`0.08291967213153839` 秒
- 平台总时间：`8.069370798766613` 秒
- 插值帧：`/root/autodl-tmp/hypercomp-data/runs/superfast-synthetic-unified/interpolated.png`
- 插值帧 SHA256：`ADD9388B0F7E0ED1D2A839852D5A2787A077B469117B3057347D94664140B789`
- 请求、worker 请求/响应、stdout、stderr 和最终结果文件均已生成

统一任务选择、模型注册、checkpoint/输入传递、隔离 Python worker、图像输出和结构化结果链路全部工作，因此组件状态提升为 `adapted`。

## 尚未完成

- 官方 `THU-HSEVI.zip` 尚未下载成功；远程 Google Drive 请求返回 HTTP 503。
- 尚未使用官方固定样例运行上游测试脚本。
- 尚未对比论文指标或官方参考输出。
- 尚未进行固定官方样例的重复运行、失败路径及干净重建验收。

完成以上官方数据验证前，不得把本记录表述为完整复现，也不得把 SuperFast 提升为 `platform_verified`。
