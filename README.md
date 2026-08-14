# 3DGS-RTMaterial V3

> Semester project for **SFU CMPT 743** (Visual Computing).

This project combines **3D Gaussian Splatting** segmentation (SAGA), **OptiX
ray tracing**, retained spherical-harmonic (SH) stylization, and a new manual
**PBR-lite** path. V3 adds dense per-Gaussian material fields, segment-level
editing, HDR lighting, Cook–Torrance GGX, shadows, and hybrid reflection /
refraction rays while deliberately leaving inverse rendering for a later stage.

---

## Demo

| RGB Baseline | Segmentation |
|:---:|:---:|
| ![RGB](demo/Screenshot%20from%202026-04-07%2022-07-32.png) | ![Segmentation](demo/Screenshot%20from%202026-04-08%2015-40-43.png) |

**Material Assignment & Ray Tracing**

| | |
|:---:|:---:|
| ![Material](demo/Screenshot%20from%202026-04-07%2022-29-49.png) | ![RT](demo/Screenshot%20from%202026-04-07%2022-38-10.png) |

**Demo Video**

[▶ Watch demo video](demo/Screencast%20from%202026-04-07%2022-43-38.webm)

---

## Features

- **Dual material pipeline**
  - *Stylized (SH Edit)* — the complete V2.2 workflow, unchanged
  - *PBR-lite* — manual albedo, roughness, metallic, opacity, and IOR
  - *Original* and *Compare Stylized/PBR* inspection modes

- **PBR-lite renderer**
  - Dense albedo / roughness / metallic / opacity storage per Gaussian
  - Segment-level UI assignment with project-state persistence
  - HDR latitude-longitude environment loading with a procedural fallback
  - Cook–Torrance GGX direct and image-based lighting
  - 3DGRT normals, depth, and opacity as the G-buffer
  - Shadow visibility and 3DGUT-style hybrid reflection / refraction rays

- **4 View Modes**
  - *RGB (Baseline)* — standard 3DGS splatting render
  - *Segmentation* — per-segment color overlay
  - *Material* — flat material-type color preview
  - *Ray-Tracing* — OptiX rendering with temporary per-segment SH/opacity edits

- **Ray-Tracing Sub-modes** (selectable in the RT panel)
  - *Stylized (SH Edit)* — SH-edited material result rendered with OptiX
  - *PBR-lite* — manual metallic-roughness shading with optional secondary rays
  - *Original* — unedited OptiX render
  - *Compare Stylized/PBR* — both material pipelines side by side

- **OptiX / 3DGRT Integration**
  - BVH built once from the loaded Gaussian model
  - Per-frame ray tracing returns RGB, surface normals, depth, and opacity
  - Visible fallback status and error reason when OptiX is unavailable
  - Depth-rasterizer normals used by the fallback path
  - Camera-space rays are transformed exactly once by 3DGRT, keeping the
    raster and OptiX viewpoints aligned
  - Adaptive interaction uses rasterization while dragging and settles to a
    full OptiX frame after release

- **Material System** — 5 presets that alter SH colour, view-dependent SH terms, and opacity

  | Material | DC/base colour | Higher-order SH | Opacity |
  |----------|----------------|-----------------|---------|
  | Default  | unchanged | unchanged | unchanged |
  | Metal    | 95% desaturated | ×1.5 | unchanged |
  | Glass    | subtle cool tint | unchanged | ×0.28 |
  | Plastic  | saturation ×1.5 | ×0.15 | unchanged |
  | Matte    | unchanged | zeroed | unchanged |

- **Segmentation**
  - Click-mode (single / multi-click) with CLIP feature similarity
  - Heatmap preview before confirming
  - HDBSCAN / MiniBatchKMeans auto-clustering
  - Visibility-aware SAM-driven 2D→3D projection (optional)
  - Lightweight cross-view instance graph: spatial Gaussian anchors, mask-to-anchor
    voting, context/SAGA/geometric evidence fusion, connectivity cuts, and
    Gaussian-level refinement restricted to instance-boundary anchors
  - Save / load segment masks

- **CLIP Material Panel** — classifies a rendered crop of the selected segment against material text prompts, or converts prompts such as `"blue glass"` and `"brushed metal"` into continuous SH/opacity parameters

- **Project State** — saves and restores the complete segmentation mask, material names/parameters, hidden segments, camera pose, and relevant UI settings in a compressed `.npz`

- **Offline Export (Stylized mode)** — immutable scene/state snapshots, background PNG/MP4/PNG-sequence
  export with progress, cancellation, tiled 4K/8K rendering and
  RGB/RGBA/depth/normals/depth-ordered ID/comparison channels

---

## Requirements

- NVIDIA GPU with a current CUDA 12.x driver (CUDA 12.8+ recommended for RTX 50-series)
- Linux (tested on Ubuntu 22.04)
- Conda / Miniconda

OptiX SDK is **not** required at runtime; `setup.sh` downloads 3DGRT, which includes the required development headers.

---

## Installation

```bash
git clone git@github.com:NocturnalHairDoc/3DGS-RTMaterial.git
cd 3DGS-RTMaterial
conda env create -f environment.yml
conda activate gaussian_splatting_v2
bash setup.sh
```

`setup.sh` does the following automatically:

1. Verifies that `gaussian_splatting_v2` is active
2. Downloads the SAGA CUDA rasterizer sources into the ignored `submodules/` directory
3. Downloads 3DGRT into the ignored `3dgrut/` directory
4. Builds the rasterizers and a PyTorch-version-compatible PyTorch3D, then installs `threedgrut` / `threedgrt_tracer`
5. Runs `runtime_check.py`, including a GPU architecture compatibility check

### Manual install (if `setup.sh` fails)

```bash
conda env create -f environment.yml
conda activate gaussian_splatting_v2

# First fetch the SAGA and 3DGRT sources, or let setup.sh do this automatically.
# 3DGS CUDA rasterizers
python patch_cuda_sources.py
pip install submodules/diff-gaussian-rasterization/ --no-build-isolation
pip install submodules/diff-gaussian-rasterization_contrastive_f/ --no-build-isolation
pip install submodules/diff-gaussian-rasterization-depth/ --no-build-isolation
pip install submodules/simple-knn/ --no-build-isolation
pip install --no-build-isolation --no-deps \
  git+https://github.com/facebookresearch/pytorch3d.git@9381c4016376345bb795b97c45a6c2de66db354a

# 3DGRT / OptiX
python patch_3dgrut_sources.py
pip install -e 3dgrut/

# Extra Python dependencies
pip install imageio einops slangtorch==1.3.18 "setuptools<80"
```

---

## Usage

```bash
conda activate gaussian_splatting_v2
python rt_gs_gui_v3.py -m /absolute/model/path --scale 1.5

# Optional HDR latitude-longitude environment
python rt_gs_gui_v3.py -m /absolute/model/path --environment /absolute/studio.hdr
```

`-m` is required and must point to a compatible trained scene directory.

The viewer expects a trained 3DGS scene with both the iteration-30000 scene
PLY and iteration-10000 contrastive feature PLY/scale gate. Model output is
intentionally not stored in Git.

### Diagnostics and tests

```bash
python runtime_check.py

# Included 61,380-Gaussian OptiX benchmark
python gpu_optix_smoke.py
```

`runtime_check.py` checks the Python dependencies, CUDA runtime, GPU
architecture and downloaded source trees. The development test suite is kept
locally and is not distributed with the repository. OptiX diagnostics require
an NVIDIA GPU and a trained scene.

### Training a scene (from scratch)

```bash
# 1. Train the base 3DGS scene
python train_scene.py -s <data_dir> -m ./output-v2/<scene_name>

# 2. Train contrastive CLIP features for segmentation
python train_contrastive_feature.py -s <data_dir> -m ./output-v2/<scene_name>
```

---

## Controls

| Action | Input |
|--------|-------|
| Orbit camera | Left-click drag |
| Pan camera | Middle-click drag |
| Zoom | Scroll wheel |
| Add segmentation prompt | Right-click on viewport |
| Confirm segment | *Confirm & hide* button |
| Roll back last segment | *Roll back* button |
| Clear all segments | *Clear all* button |
| Save/load complete editing state | *Project & Export → Save/Load project* |
| Export high-resolution still | *Project & Export → Export PNG* |
| Export 360° turntable | *Project & Export → Export turntable MP4* |

---

## Project Structure

```
3DGS-RTMaterial-V3/
├── rt_gs_gui_v3.py            # V3 dual Stylized/PBR-lite viewer (entry point)
├── pbr_lite.py                # PBR fields, HDR, GGX and hybrid compositor
├── pbr_gpu_smoke.py           # Real-scene multi-round GPU benchmark
├── compare_pbr_versions.py    # V2.2/V3 report generator
├── rt_gs_gui_sh_clip.py       # Retained V2.2 SH-material + CLIP viewer
├── rt_gs_gui.py               # Shared base viewer and Blinn-Phong alternative
├── saga_gui.py                # Original SAGA GUI (segmentation only)
├── optix_integration/         # OptiX / 3DGRT integration module
│   ├── optix_renderer.py      #   BVH build + per-frame ray trace
│   ├── gaussian_adapter.py    #   Wraps SAGA GaussianModel for 3DGRT API
│   ├── ray_generator.py       #   Camera → ray batch conversion
│   ├── material_compositor.py #   Blinn-Phong shading on ray-traced normals
│   └── build_plugin.py        #   Compiles the 3DGRT Slang/CUDA plugin
├── material_sh_edit/          # SH-based material editor utilities
├── render_policy.py           # Interactive raster/OptiX and camera policy
├── export_manager.py          # Background export worker and status queue
├── segmentation_utils.py      # Cross-view mask association and visibility helpers
├── training_utils.py          # NaN-safe contrastive loss helpers
├── benchmarks/                # Reproducible 61k GPU smoke results
├── project_state.py           # Versioned state persistence and migration
├── undo_manager.py            # Bounded lightweight undo/redo history
├── 3dgrut/                    # NVIDIA 3DGRT source downloaded by setup.sh
├── scene/                     # Scene + GaussianModel classes
├── gaussian_renderer/         # 3DGS CUDA rasterizer wrappers
├── submodules/                # downloaded by setup.sh; not stored in Git
├── demo/                      # Screenshots and demo video
├── runtime_check.py           # dependency/GPU architecture diagnostics
├── environment.yml            # Conda environment specification
└── setup.sh                   # One-shot environment setup
```

---

## Known Issues / Limitations

- The 3DGRT OptiX plugin compiles Slang/CUDA kernels on **first run** — this takes 1–3 minutes. Subsequent runs use the cached binary.
- 3DGRT commit `a37ef721` requires the Slang pointer-access API; pin SlangTorch to `1.3.18`.
- `setuptools >= 80` removes `pkg_resources`; pin to `< 80`.
- Python 3.10 is required by this environment.
- Interactive RT intentionally uses a rasterized material preview during camera drag; a full OptiX frame replaces it on release.
- PBR values are manually assigned; V3 does not recover intrinsic materials or
  illumination from training images and is not a full inverse-rendering system.
- 3DGRT Gaussian intersection normals can expose ellipse/splat contours in dense
  scenes. Depth-tangent fusion reduces the artifact but does not equal trained,
  multi-view-consistent normals.
- Secondary rays are single-bounce hybrid rays and shadows use one visibility
  ray. Rough transmission is an approximation rather than spectral volume tracing.
- Background export currently renders the retained Stylized SH pipeline; PBR-lite
  export is planned for the next renderer/export integration pass.
- SAM-driven fusion uses a projected-center z-buffer rather than full Gaussian
  coverage; thin structures and heavy occlusion can still fragment or merge.
- Segmentation errors and overlapping Gaussians can cause material bleeding at object boundaries.
- Trained scenes are not included in the repository.
- Changing hidden segments rebuilds the OptiX BVH once; subsequent frames reuse the filtered BVH. Large scenes may briefly pause on the first frame after a visibility change.
- MP4 dimensions must be even. Very large exports still require enough VRAM for one tile and the scene BVH.
- A frozen export duplicates render-critical scene tensors on the GPU so an
  in-flight export cannot change when the GUI is edited; very large scenes may
  need CPU staging in a future release.

### Interactive preview setting

Adaptive raster preview can be disabled when profiling full OptiX interaction:

```bash
export RTM_ADAPTIVE_RT_PREVIEW=0
```

The same setting is available in the Ray-Tracing panel. The corrected OptiX
camera transform is now the only supported camera path.

---

## Acknowledgements

- [SAGA — Segment Any 3D GAussians](https://github.com/Jumpat/SegAnyGAussians)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [NVIDIA 3DGRT / threedgrut](https://github.com/nv-tlabs/3dgrut)
- [CLIP](https://github.com/openai/CLIP)

## License

Project-authored code is available under Apache-2.0. Derived research
components retain their upstream terms, including the non-commercial
restrictions in the original 3DGS source headers. See `LICENSE.md` for details.
