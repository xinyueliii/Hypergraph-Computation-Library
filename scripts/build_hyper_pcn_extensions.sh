#!/usr/bin/env bash
set -eu

project_root=${1:-/root/autodl-tmp/Hypergraph-Computation-Library}
python_root=${2:-/root/autodl-tmp/hypercomp-data/envs/visual-cu121}
cuda_root=${3:-/root/autodl-tmp/hypercomp-data/toolchains/cuda-12.1}
source_root="$project_root/third_party/Hyper-PCN"

"$python_root/bin/python" "$project_root/scripts/prepare_hyper_pcn_source.py" \
  --source-root "$source_root"

export CUDA_HOME="$cuda_root"
export PATH="$python_root/bin:$cuda_root/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=${MAX_JOBS:-4}
site_packages="$python_root/lib/python3.11/site-packages"
export CPATH="$site_packages/nvidia/cusparse/include:$site_packages/nvidia/cusolver/include:$site_packages/nvidia/cublas/include"

"$python_root/bin/pip" install --no-build-isolation "$source_root/extensions/chamfer_dist"
"$python_root/bin/pip" install --no-build-isolation "$source_root/extensions/pointnet2_ops_lib"
