# SuperFast 本机环境预评估（2026-08-10）

## 结论

SuperFast 已保留统一适配器和独立 worker，但当前本机不适合继续强行完成官方推理，组件仍为 `source_audited`。建议在带完整 CUDA 编译工具链的 Linux GPU 服务器上完成 slayerPytorch、checkpoint 和数据样例验证。

## 已完成

- 固定 SuperFast 源码提交 `b4d29efc75c4a6ca07de4c5f56f555de48b65741`。
- 确认主模型通过 `model/EventEncoder.py` 导入 `slayerSNN`。
- 确认 slayerPytorch 需要编译 C++/CUDA 源码，上游声明的测试平台为 Ubuntu、GCC 和 CUDA 环境。
- 创建隔离基础环境 `D:\Anaconda\envs\hypercomp-superfast`，Python 3.8.20。
- 提交目标环境规格 `environments/superfast-win-cu111.yml`。

## 当前阻塞

- 本机 `nvcc` 不在 PATH 中，PyTorch 自带 CUDA runtime 不能替代 CUDA 源码编译器。
- slayerPytorch 上游没有声明 Windows 为受支持测试平台，并且需要 C++/CUDA 编译。
- 旧版 PyTorch pip 索引在本机出现 SSL 证书读取错误；Conda 旧依赖求解长时间无结果后被主动终止。基础 Python 3.8 环境未受损，但尚未安装 PyTorch。
- 官方 checkpoint 和最小 THU-HSEVI/HS-ERGB 场景尚未下载，因此不存在端到端运行证据。
- 官方代码面向高分辨率事件体、相邻帧和多个网络分支，RTX 3060 6 GB 的真实峰值显存仍未知。

## 环境与可行性决定

- 环境 ID：`superfast-win-cu111-target`。
- 本机结论：`remote_recommended`。
- 建议服务器：Linux、可用 `nvcc` 和 C++ 编译工具链、CUDA 11.x 兼容驱动，优先 12 GB 或更高显存。
- 在服务器上应先编译固定版本 slayerPytorch，再下载一个官方 checkpoint 和一个最小合法场景，通过原项目路径后再运行 HyperComp worker。
