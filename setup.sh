#!/usr/bin/env bash
# setup.sh — one-shot environment setup for 3DGS-RTMaterial
#
# Prerequisites:
#   - Miniconda / Anaconda installed
#   - NVIDIA GPU with CUDA 12.x driver
#   - OptiX SDK not required at runtime (stubs bundled in 3dgrut/)
#
# Usage:
#   bash setup.sh

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] Creating conda environment from environment.yml ==="
conda env create -f "$REPO_DIR/environment.yml"
# Activate for subsequent pip installs
eval "$(conda shell.bash hook)"
conda activate gaussian_splatting

echo "=== [2/4] Installing 3DGS CUDA submodules ==="
pip install "$REPO_DIR/submodules/diff-gaussian-rasterization/" --no-build-isolation
pip install "$REPO_DIR/submodules/diff-gaussian-rasterization_contrastive_f/" --no-build-isolation
pip install "$REPO_DIR/submodules/diff-gaussian-rasterization-depth/" --no-build-isolation
pip install "$REPO_DIR/submodules/simple-knn/" --no-build-isolation

echo "=== [3/4] Installing 3DGRT / OptiX packages ==="
# threedgrut: Python training/rendering framework
pip install -e "$REPO_DIR/3dgrut/"
# threedgrt_tracer: OptiX ray-tracing plugin (compiles CUDA/Slang kernels on first run)
pip install -e "$REPO_DIR/3dgrut/threedgrt_tracer/"

echo "=== [4/4] Done ==="
echo ""
echo "Activate the environment with:"
echo "  conda activate gaussian_splatting"
echo ""
echo "Run the viewer with:"
echo "  python rt_gs_gui.py -m ./output/<scene>"
