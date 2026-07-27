# 3DGS-RTMaterial for Windows

This directory contains the native Windows implementation of 3DGS-RTMaterial. It is
self-contained and does not modify the Linux implementation in the repository root.

## Requirements

- Windows 10 or Windows 11, x64
- NVIDIA GPU and current NVIDIA driver
- Visual Studio 2022 Build Tools or newer, with:
  - Desktop development with C++
  - Windows 10/11 SDK
- Miniconda or Anaconda
- PowerShell 5.1 or newer

The setup script creates a Python 3.11 environment, installs the CUDA 12.8 toolkit,
installs PyTorch with CUDA support, builds the Gaussian rasterizers, and optionally
prepares the 3DGRUT/OptiX integration.

## Installation

Open PowerShell in the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd .\v_windows
.\setup_windows.ps1
```

To install only the rasterization, segmentation, and material preview components:

```powershell
.\setup_windows.ps1 -SkipOptix
```

Downloaded source archives and third-party dependencies are stored under `vendor`.
Model checkpoints are stored under `checkpoints`. Both directories are excluded from
version control.

## Environment check

```powershell
conda activate 3dgs_rtmaterial_windows
python .\diagnose.py
```

If Conda activation reports a missing VS2017 or MSVC 14.16 toolset, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\fix_conda_activation.ps1
```

Close the terminal, open a new PowerShell window, and activate the environment again.

## Dataset layout

The training commands expect a COLMAP dataset with the following layout:

```text
<dataset>
├── images
├── images_4
└── sparse
    └── 0
```

The examples below use PowerShell variables so paths can be changed in one place:

```powershell
$Dataset = "D:\datasets\garden"
$Model = ".\output\garden"
```

## Train the scene

```powershell
python launcher.py train-scene `
  -s $Dataset `
  -m $Model `
  -i images_4 `
  --iterations 30000 `
  --save_iterations 7000 30000 `
  --checkpoint_iterations 1000 5000 10000 20000 30000
```

Render the trained scene:

```powershell
python launcher.py render `
  -m $Model `
  --iteration 30000 `
  --target scene
```

## Extract SAM 2 masks

The default setup installs SAM 2.1 Hiera Large and downloads its checkpoint. To install
or replace it separately:

```powershell
.\install_sam2.ps1 -ModelSize large
```

Generate SAGA-compatible masks from `images_4`:

```powershell
python launcher.py extract-sam2 `
  --image_root $Dataset `
  --downsample 4
```

Existing mask files are skipped. Use `--limit 1` for a quick check or `--overwrite` to
replace existing files. Results are written to `<dataset>\sam_masks`.

## Generate mask scales

```powershell
python launcher.py get-scale `
  -m $Model `
  --iteration 30000
```

Results are written to `<dataset>\mask_scales`. Existing files are skipped. Use
`--overwrite` to recompute them, `--limit 1` to process one view, or `--idx N` to process
a specific training view.

When the optional depth rasterizer is unavailable, the Windows renderer computes the
same accumulated camera-space depth with the standard SAGA rasterizer.

## Train contrastive features

```powershell
python launcher.py train-feature `
  -s $Dataset `
  -m $Model `
  --iteration 30000 `
  --iterations 10000 `
  --num_sampled_rays 512
```

Large point clouds use an exact SciPy `cKDTree` instead of allocating a full pairwise
distance matrix. The first run writes a reusable KNN cache next to the scene point cloud.
Later runs with the same scene and `--smooth_K` value load that cache directly.

## Launch the viewer

After scene and feature training:

```powershell
.\run_viewer.ps1 -ModelPath $Model
```

The equivalent launcher command is:

```powershell
python launcher.py viewer -m $Model -s 30000 -f 10000
```

The model directory should contain:

```text
point_cloud\iteration_30000\scene_point_cloud.ply
point_cloud\iteration_10000\contrastive_feature_point_cloud.ply
point_cloud\iteration_10000\scale_gate.pt
```

## Command dispatcher

`launcher.py` initializes the Windows DLL search path and exposes these commands:

| Command | Purpose |
| --- | --- |
| `viewer` | Launch the material viewer |
| `saga` | Launch the SAGA viewer |
| `train-scene` | Train the base Gaussian scene |
| `train-feature` | Train contrastive features |
| `render` | Render trained views |
| `convert` | Run dataset conversion |
| `extract-sam2` | Generate SAM 2 masks |
| `get-scale` | Generate per-mask scales |

## Generated directories

The following directories are local runtime data and are not committed:

- `vendor`: downloaded third-party source and build trees
- `checkpoints`: SAM 2 and other model weights
- `output`: trained models, KNN caches, renders, and checkpoints
- `__pycache__`, `build`, and extension caches
