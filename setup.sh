#!/usr/bin/env bash
# Reproducible environment/bootstrap installer for 3DGS-RTMaterial.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGA_URL="${SAGA_URL:-https://github.com/Jumpat/SegAnyGAussians.git}"
THREEDGRUT_URL="${THREEDGRUT_URL:-https://github.com/nv-tlabs/3dgrut.git}"
SAGA_COMMIT="${SAGA_COMMIT:-2d4c5d77c857c956d747e4775d3d72c4ec5dfe16}"
THREEDGRUT_COMMIT="${THREEDGRUT_COMMIT:-a37ef721012dea0f29c0fcfff2d525023b4e854a}"
PYTORCH3D_COMMIT="${PYTORCH3D_COMMIT:-9381c4016376345bb795b97c45a6c2de66db354a}"

command -v conda >/dev/null || { echo "Error: conda is required." >&2; exit 1; }
command -v git >/dev/null || { echo "Error: git is required." >&2; exit 1; }

if [[ "${CONDA_DEFAULT_ENV:-}" != "gaussian_splatting_v2" ]]; then
    echo "Error: activate gaussian_splatting_v2 before running setup.sh." >&2
    exit 1
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_DIR/.cache/torch_extensions/$CONDA_DEFAULT_ENV}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_DIR/.cache/$CONDA_DEFAULT_ENV}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
echo "=== [1/5] Using isolated conda environment: $CONDA_PREFIX ==="
echo "Extension cache: $TORCH_EXTENSIONS_DIR"

# A conda libtorch plus a newer pip torchvision can leave two incompatible
# torch distributions in one prefix. Repair only when the imported files do
# not match the requested wheels; healthy reruns avoid a multi-GB download.
if ! python -c 'import torch, torchvision; assert torch.__version__.startswith("2.10."); assert torchvision.__version__.startswith("0.25.")'; then
    python -m pip install --force-reinstall --no-cache-dir "torch==2.10.0" "torchvision==0.25.0"
fi
python -m pip install "imageio-ffmpeg==0.6.0" "slangtorch==1.3.18"

echo "=== [2/5] Fetching SAGA CUDA rasterizer sources ==="
if [[ ! -f "$REPO_DIR/submodules/diff-gaussian-rasterization/setup.py" ]]; then
    bootstrap_dir="$(mktemp -d /tmp/3dgs-rtm-saga.XXXXXX)"
    git clone --no-checkout "$SAGA_URL" "$bootstrap_dir/saga"
    git -C "$bootstrap_dir/saga" checkout --detach "$SAGA_COMMIT"
    mkdir -p "$REPO_DIR/submodules"
    cp -a "$bootstrap_dir/saga/submodules/." "$REPO_DIR/submodules/"
fi

echo "=== [3/5] Installing 3DGS CUDA rasterizers ==="
python "$REPO_DIR/tools/setup/patch_cuda_sources.py"
for package in \
    diff-gaussian-rasterization \
    diff-gaussian-rasterization_contrastive_f \
    diff-gaussian-rasterization-depth \
    simple-knn; do
    python -m pip install "$REPO_DIR/submodules/$package/" --no-build-isolation
done
python -m pip install --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/pytorch3d.git@$PYTORCH3D_COMMIT"

echo "=== [4/5] Fetching and installing 3DGRT ==="
if [[ ! -f "$REPO_DIR/3dgrut/setup.py" ]]; then
    bootstrap_dir="$(mktemp -d /tmp/3dgs-rtm-3dgrut.XXXXXX)"
    git clone --no-checkout "$THREEDGRUT_URL" "$bootstrap_dir/3dgrut"
    git -C "$bootstrap_dir/3dgrut" checkout --detach "$THREEDGRUT_COMMIT"
    mkdir -p "$REPO_DIR/3dgrut"
    cp -a "$bootstrap_dir/3dgrut/." "$REPO_DIR/3dgrut/"
fi
if [[ ! -f "$REPO_DIR/3dgrut/threedgrt_tracer/dependencies/optix-dev/include/optix.h" ]]; then
    git -C "$REPO_DIR/3dgrut" submodule update --init --depth 1 \
        threedgrt_tracer/dependencies/optix-dev
fi
python "$REPO_DIR/tools/setup/patch_3dgrut_sources.py"
python -m pip install -e "$REPO_DIR/3dgrut/"

echo "=== [5/5] Verifying runtime ==="
python "$REPO_DIR/tools/runtime_check.py"

echo
echo "Installation complete. Activate with: conda activate gaussian_splatting_v2"
echo "Run V3 with: python rt_gs_gui_v3.py -m /absolute/model/path"
