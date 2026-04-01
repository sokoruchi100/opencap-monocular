#!/bin/bash
# Build and install pytorch3d from source.
#
# Environment requirements (verified):
#   conda env : opencap-mono-slim
#   PyTorch   : 2.1.2+cu118
#   Python    : 3.9
#   CUDA home : /usr/local/cuda-11.8  (matches PyTorch's build CUDA version)
#   nvcc      : /usr/local/cuda-11.8/bin/nvcc
#
# Must be run with the conda env active:
#   conda activate opencap-mono-slim && bash build_pytorch3d.sh

set -e  # exit on first error

# Point to CUDA 11.8 to match PyTorch's build — avoids ABI mismatch with
# the system default nvcc (12.0)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Force CUDA extensions to be compiled (skip CPU-only fallback)
export FORCE_CUDA=1

# Target only the GPU arch(s) present on this machine to speed up compilation.
# Check yours with: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Common values: 7.5 (Turing/RTX 20xx), 8.0 (A100), 8.6 (Ampere/RTX 30xx), 8.9 (RTX 40xx)
# Leave unset to compile for all supported archs (slower but more portable).
export TORCH_CUDA_ARCH_LIST="8.9" # CCI Server

echo "=== CUDA_HOME : $CUDA_HOME"
echo "=== nvcc      : $(nvcc --version | grep release)"
echo "=== python    : $(python --version)"
echo "=== torch     : $(python -c 'import torch; print(torch.__version__)')"
echo ""

# --no-build-isolation lets setup.py see the conda env's torch at build time.
# Without it, pip creates a clean subprocess where torch isn't found.
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo ""
echo "=== Verifying install ==="
python -c "import pytorch3d; print('pytorch3d version:', pytorch3d.__version__)"
