# HyperComp 运行环境与本机推理能力映射

更新日期：2026-08-11。

## 本机推理基线

- GPU：NVIDIA GeForce RTX 3060 Laptop GPU，6 GB 显存。
- 驱动：566.03。
- 2026-08-10 空闲状态：约 5996 MiB 可用显存，48 摄氏度，GPU 利用率 0%。
- D 盘：总容量约 514.4 GB，可用约 232.2 GB；环境、checkpoint 和大体积运行产物优先放在 D 盘。
- 系统 CUDA 编译器：`nvcc` 不在 PATH 中。带自定义 CUDA 源码编译的项目不能仅靠 PyTorch 自带 CUDA runtime 解决。

推理能力结论采用三档：

- `local_safe`：可用 batch 1 和官方小模型在本机安全推理。
- `local_conditional`：需要降低分辨率、精度或规模，或关闭其他 GPU 程序。
- `remote_recommended`：本机缺少必要工具链，或预计显存/系统兼容风险明显，优先使用 Linux GPU 服务器。

## 模型—环境映射

| 环境 ID | 本地位置 | 关键版本 | 分配组件 | 当前状态 |
|---|---|---|---|---|
| `core-cu126` | 已有核心环境 | Python 3.10.16；PyTorch 2.7.0+cu126 | DHG、HGNN、HGNN+ 的旧验证环境 | HGNN/HGNN+ CPU/GPU 已运行；保留为兼容备选 |
| `shared-win-cu121` | `D:\Anaconda\envs\hypercomp-yolov13` | Python 3.11.15；PyTorch 2.2.2+cu121；Torchvision 0.17.2+cu121；NumPy 1.26.4；SciPy 1.17.1；scikit-learn 1.6.1；Optuna 4.5.0；OpenCV 4.10.0；einops 0.8.1 | DHG/HGNN、YOLOv13、Hyper-YOLO、SoftHGNN | 外部三模型均已完成 GPU 统一入口运行，且 HGNN/HGNN+ 回归测试通过；SoftHGNN 仅为未训练 smoke |
| `visual-remote-cu121` | `/root/autodl-tmp/hypercomp-data/envs/visual-cu121` | Ubuntu 20.04；Python 3.11.15；PyTorch 2.2.2+cu121；Torchvision 0.17.2+cu121；NumPy 1.26.4 | DHG/HGNN、YOLOv13、Hyper-YOLO、SoftHGNN、Hyper-PCN、PVRNet | PVRNet 官方权重固定样例和 Hyper-PCN 真实网格衍生局部点云均已通过统一入口；`pip check` 通过 |
| `count-anything-remote-cu126` | `/root/autodl-tmp/hypercomp-data/envs/count-anything-py311` | Ubuntu 20.04；Python 3.11.15；PyTorch 2.7.1+cu126；Torchvision 0.22.1+cu126；NumPy 1.26.4 | Count Anything | 官方 checkpoint 单图文本计数通过；因共享环境缺少 `torch.nn.attention` 才拆分；`decord` 仅有平台元数据告警，实际导入与推理通过 |
| `superfast-win-cu111-target` | `D:\Anaconda\envs\hypercomp-superfast` | 已创建 Python 3.8.20 基础环境；目标为 PyTorch 1.9.x、Torchvision 0.10.x、CUDA 11.1、slayerPytorch | SuperFast 本机备选 | 本机无 nvcc，未继续构建 |
| `hypercomp-remote-core-py310` | `/root/autodl-tmp/hypercomp-data/envs/core-py310` | Python 3.10.20 | 远程任务注册、CLI 与独立 worker 调度 | `tasks` 和 SuperFast 模型查询通过；统一入口运行通过 |
| `superfast-remote-cu113` | AutoDL `/root/miniconda3` | Ubuntu 20.04；Python 3.8.10；PyTorch 1.10.0+cu113；Torchvision 0.11.1+cu113；CUDA toolkit 11.3；GCC 9；slayerPytorch `01beeeb6` | SuperFast worker | slayer CUDA 调用、官方 checkpoint 加载、上游 API 与统一入口合成样例推理均通过 |
| `event-remote-cu113` | `/root/autodl-tmp/hypercomp-data/envs/event-cu113` | Ubuntu 20.04；Python 3.8.10；PyTorch 1.10.0+cu113；Torchvision 0.11.1+cu113；CUDA toolkit 11.3；GCC 9；PyMeshLab 2021.10；Rich 13.9.4 | E-HRSAI、E-3DTrack、MeshNet | E-HRSAI 已通过官方 checkpoint 合成结构统一入口；MeshNet 已通过替代 checkpoint 与官方格式单网格统一入口；E-3DTrack 已用官方 checkpoint 和官方测试序列通过统一入口；`pip check` 通过 |
| `cuda-12.1-toolchain` | `/root/autodl-tmp/hypercomp-data/toolchains/cuda-12.1` | CUDA compiler 12.1.105；约 184 MB | 供共享环境编译兼容扩展 | 已用于 visual-cu121 中 Hyper-PCN 的 Chamfer 和 PointNet2 Ops；不是独立 Python 环境 |

远程服务器的推荐环境矩阵与硬件采购规格见 [`docs/服务器租用与优先方法环境评估.md`](服务器租用与优先方法环境评估.md)。AutoDL RTX 3090 24 GB、Ubuntu 20.04 是当前默认运行和验收平台。PVRNet 与 Hyper-PCN 复用 `visual-remote-cu121`；Count Anything 仅因实测 PyTorch API 冲突保留独立 cu126 环境；E-HRSAI、E-3DTrack 和 MeshNet 共享 `event-remote-cu113`；SuperFast 保留已验证旧版运行时。

SuperFast 外部资产位于 `/root/autodl-tmp/hypercomp-data`：checkpoint 为 `checkpoints/superfast/ckpt_THU_HSEVI.pth`，大小 `418,931,023` bytes，SHA256 为 `4828b7647b77198795165d2a3c18c37c2773b4c785fd06a96da9df296965ce88`。合成样例、缓存和运行产物同样保存在该数据根目录，不进入 Git。

现代视觉远程环境在补充 Hyper-PCN 依赖后大小约 5.8 GB。早期环境 freeze 保存于 `runs/remote-visual-20260811-environment-freeze.txt`，SHA256 为 `ba8b7e33d0aeb56d65240087a986823def10152e55d5e70243624f4cce395365`；该 freeze 早于 Hyper-PCN 扩展。完整记录见 `docs/validation/remote-shared-visual-2026-08-11.md` 和 `docs/validation/hyper-pcn-remote-adaptation-2026-08-11.md`。

## 组件本机可行性

| 组件 | 结论 | 依据与限制 |
|---|---|---|
| YOLOv13-N | `local_safe` | 已实际完成单图 GPU 推理；6 GB 显存足够。 |
| Hyper-YOLO-N | `local_safe` | 已实际完成单图 GPU 推理；实测 PyTorch 峰值分配约 185.5 MB。 |
| SoftHGNN CIFAR 模型 | `local_safe` | 默认模型约 1882 万参数；batch 1、32×32 前向实测峰值约 90.3 MB。无公开训练权重。 |
| SuperFast | `remote_recommended` | 本机仍缺少 nvcc；已在远程 RTX 3090 完成 slayer 编译和带权重合成样例推理，最大分配显存约 910 MiB。官方数据分辨率与完整任务负载仍未验证。 |

## 环境管理规则

1. 默认只在 AutoDL 远程运行新组件；本机保留 Git 工作副本和无依赖检查。
2. 新组件先在 `visual-cu121` 与 `event-cu113` 之间做兼容性分配，不主动升级或降级共享环境的 PyTorch/CUDA。
3. 纯 Python 兼容依赖可以加入共享环境；修改后必须运行 `pip check` 和已分配模型的聚焦回归，并同步环境规格。
4. 包名相同但源码不同的研究项目，可使用同一环境但通过独立工作进程和各自源码目录加载。worker 不等于独立环境。
5. 仅缺 `nvcc` 时优先复用共享编译工具链；只有复现了 Python、PyTorch/CUDA、ABI、二进制扩展或系统库冲突后才建立完整新环境。
6. 远程环境、checkpoint、样例、缓存和运行产物统一放在 `/root/autodl-tmp/hypercomp-data`，不进入 Git；大下载前检查空间。
