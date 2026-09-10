# 已有本地模型的远程共享环境验证记录

日期：2026-08-11

覆盖组件：HGNN、HGNN+（代码名 `HGNNP`）、YOLOv13-N、Hyper-YOLO-N、SoftHGNN Image Classification。

本记录证明上述模型能够在 AutoDL RTX 3090 服务器上通过 HyperComp 统一入口运行。检测结果来自官方 checkpoint 和固定图片；HGNN/HGNN+ 使用确定性微型超图；SoftHGNN 使用确定性随机初始化 smoke。所有结果均为工程适配证据，不是论文数据集或精度复现。

## 服务器与共享环境

- 系统：Ubuntu 20.04，glibc 2.31
- GPU：NVIDIA GeForce RTX 3090，24 GB
- 驱动：570.124.04
- 环境 ID：`visual-remote-cu121`
- 环境路径：`/root/autodl-tmp/hypercomp-data/envs/visual-cu121`
- Python：3.11.15
- PyTorch：2.2.2+cu121
- torchvision：0.17.2+cu121
- CUDA runtime：12.1
- NumPy：1.26.4
- SciPy：1.17.1
- OpenCV：4.10.0 headless
- einops：0.8.1
- `pip check`：`No broken requirements found.`
- 环境大小：约 5.7 GB
- 环境 freeze：`/root/autodl-tmp/hypercomp-data/runs/remote-visual-20260811-environment-freeze.txt`
- freeze SHA256：`ba8b7e33d0aeb56d65240087a986823def10152e55d5e70243624f4cce395365`
- 可复现规格：`environments/visual-remote-cu121.yml`
- 远程回归：组件注册 `5/5`、原生节点分类 `3/3` unittest 通过

安装过程中，服务器默认 PyPI 镜像下载部分 wheel 过慢。已完成的 PyTorch/CUDA wheel 被保存到 `/root/autodl-tmp/hypercomp-data/cache/visual-cu121-wheelhouse` 后离线安装，剩余依赖从清华镜像补齐。该 wheelhouse 是服务器缓存，不进入 Git。SuperFast 的 Python 3.8/PyTorch 1.10 环境没有被修改。

## 固定源码与输入

| 组件 | 固定源码 commit | 远程源码目录 |
|---|---|---|
| YOLOv13 | `73289949533efac82bb5f72ec19b746618656bd2` | `third_party/YOLOv13` |
| Hyper-YOLO | `9bfdabd8b97b5ee5da04e5df30d140a9d15557c5` | `third_party/Hyper-YOLO` |
| SoftHGNN | `3f46b20eeb226616a8af89f321b2e1dc3ee3634f` | `third_party/SoftHGNN` |

外部源码由本地已验证 commit 的 `git archive` 部署，不保留嵌套 `.git`。检测固定输入为 YOLOv13 源码中的 `bus.jpg`，大小 137,419 bytes，SHA256 为 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`。

检测权重：

- YOLOv13-N：`yolov13n.pt`，SHA256 `6653035017b0f111f80ec11ed914874ea85699b104aeac1e46e517d16889d6b7`
- Hyper-YOLO-N：`hyper-yolon.pt`，SHA256 `c51808f9b019c136ecd47ab50e59ac3e93ccf41de2f54e11b70542a217bc8932`

## HGNN 与 HGNN+

两个原生模型均使用 `cuda:0` 完成 2 epoch smoke，并使用产生的 checkpoint 再执行 evaluate。

| 模型 | smoke | checkpoint restore | smoke 总耗时 | restore 总耗时 | checkpoint SHA256 |
|---|---|---|---:|---:|---|
| HGNN | `succeeded` | `succeeded` | 1.053 s | 0.466 s | `9223abd3486824476240de77e1ec761063e5932e9a884ab06a24e1c57786d452` |
| HGNN+ | `succeeded` | `succeeded` | 0.999 s | 0.503 s | `c2e44889402556fb25f979ac7cef6f8723b203f19420b04d97257df0604cfdb3` |

运行目录：

- `/root/autodl-tmp/hypercomp-data/runs/hgnn-remote-20260811`
- `/root/autodl-tmp/hypercomp-data/runs/hgnn-remote-restore-20260811`
- `/root/autodl-tmp/hypercomp-data/runs/hgnnp-remote-20260811`
- `/root/autodl-tmp/hypercomp-data/runs/hgnnp-remote-restore-20260811`

微型样例只有 12 个节点，验证目标是 GPU 消息传播、训练、产物写入和 checkpoint 恢复，不用于衡量论文精度。

## YOLOv13-N

- 统一入口状态：`succeeded`
- 检测：5 个 person、1 个 bus
- worker 时间：1.520 s
- 平台总时间：4.167 s
- stderr：空
- 渲染图片 SHA256：`613be8b3f8a36f15ed135d22cf683864a78c386a6ab54fc6848a7dd80b810a80`
- 运行目录：`/root/autodl-tmp/hypercomp-data/runs/yolov13-remote-20260811`

类别与数量和本地固定样例结果一致。浮点框坐标与渲染图片哈希不要求跨 GPU/操作系统逐字节相同。

## Hyper-YOLO-N

- 统一入口状态：`succeeded`
- 检测：4 个 person、1 个 bus、1 个 skateboard
- worker 时间：1.472 s
- 平台总时间：4.197 s
- stderr：空
- 渲染图片 SHA256：`ca8d591dde069d48241b70301313f4296726d59251455aa3ccb6d07ba2dcd435`
- 运行目录：`/root/autodl-tmp/hypercomp-data/runs/hyper-yolo-remote-20260811`

检测类别、数量和渲染图片哈希均与本地记录一致。

## SoftHGNN

- 动作：`smoke`
- 统一入口状态：`succeeded`
- 输入：同一张 `bus.jpg`，缩放为 CIFAR-10 的 32 x 32 输入
- checkpoint：未加载
- worker GPU 前向：0.180 s
- 平台总时间：2.840 s
- stderr：空
- 运行目录：`/root/autodl-tmp/hypercomp-data/runs/soft-hgnn-remote-20260811`

该运行只证明模型构建、图片预处理、GPU 前向和统一结果结构工作。随机初始化 top-k 不具有分类意义；得到训练 checkpoint 前不得表述为有效分类推理。

## 状态边界

五个模型原有 `adapted` 状态保持不变。本次增加了 Linux RTX 3090 的第二套运行证据，但尚未执行完整干净重建、重复验收、正式数据集或所有失败路径，因此不统一提升为 `platform_verified`。

远程回归最初暴露出测试中硬编码 `RTX 3060` 的可移植性问题。断言已改为比较结果记录与 `torch.cuda.get_device_name(0)`，随后在 RTX 3090 上通过；该修改不改变模型行为。
