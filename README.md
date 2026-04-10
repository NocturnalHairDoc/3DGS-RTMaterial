# 3DGS-RTMaterial

> Semester project for **SFU CMPT 743** (Visual Computing).

An interactive viewer that combines **3D Gaussian Splatting** segmentation (SAGA) with **OptiX ray-traced normals** and **per-segment Blinn-Phong material shading**.  Segments are assigned physical material types (Metal, Glass, Plastic, Matte) and rendered with physically-motivated BRDFs driven by true ray-traced surface normals from NVIDIA's 3DGRT tracer.

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

- **4 View Modes**
  - *RGB (Baseline)* — standard 3DGS splatting render
  - *Segmentation* — per-segment color overlay
  - *Material* — flat material-type color preview
  - *Ray-Tracing* — full Blinn-Phong shading with OptiX normals

- **Ray-Tracing Sub-modes** (selectable in the RT panel)
  - *Material* — per-segment physically shaded render
  - *Depth* — false-color depth visualization
  - *Normals* — world-space surface normal visualization

- **OptiX / 3DGRT Integration**
  - BVH built once from the loaded Gaussian model
  - Per-frame ray tracing returns RGB, surface normals, depth, and opacity
  - Graceful fallback to depth-rasterizer normals when OptiX is unavailable

- **Material System** — 5 material types with independent Blinn-Phong parameters

  | Material | Ambient | Diffuse | Specular | Shininess | Albedo |
  |----------|---------|---------|----------|-----------|--------|
  | Default  | 0.10 | 0.70 | 0.20 |  16 | scene SH color |
  | Metal    | 0.05 | 0.15 | 1.50 | 128 | silver-grey |
  | Glass    | 0.05 | 0.08 | 1.80 | 256 | sky blue |
  | Plastic  | 0.10 | 0.70 | 0.50 |  32 | red |
  | Matte    | 0.12 | 1.00 | 0.00 |   1 | warm beige |

- **Segmentation**
  - Click-mode (single / multi-click) with CLIP feature similarity
  - Heatmap preview before confirming
  - HDBSCAN / MiniBatchKMeans auto-clustering
  - SAM-driven 2D→3D projection (optional)
  - Save / load segment masks

- **CLIP Material Detection** — enter a text prompt (e.g. `"shiny gold"`) to auto-assign a material type to the selected segment

- **Configurable Lighting** — azimuth and elevation sliders control the directional light in world space

---

## Requirements

- NVIDIA GPU with CUDA 12.x driver
- Linux (tested on Ubuntu 22.04)
- Conda / Miniconda

OptiX SDK is **not** required at runtime — stubs are bundled inside `3dgrut/`.

---

## Installation

```bash
git clone <this-repo> --recurse-submodules
cd 3DGS-RTMaterial
bash setup.sh
```

`setup.sh` does the following automatically:

1. Creates the `gaussian_splatting` conda environment from `environment.yml`
2. Builds the 3DGS CUDA rasterizer submodules (`diff-gaussian-rasterization`, `simple-knn`, …)
3. Installs the `threedgrut` and `threedgrt_tracer` packages from `3dgrut/`

### Manual install (if `setup.sh` fails)

```bash
conda env create -f environment.yml
conda activate gaussian_splatting

# 3DGS CUDA submodules
pip install submodules/diff-gaussian-rasterization/ --no-build-isolation
pip install submodules/diff-gaussian-rasterization_contrastive_f/ --no-build-isolation
pip install submodules/diff-gaussian-rasterization-depth/ --no-build-isolation
pip install submodules/simple-knn/ --no-build-isolation

# 3DGRT / OptiX
pip install -e 3dgrut/
pip install -e 3dgrut/threedgrt_tracer/

# Extra Python dependencies
pip install imageio einops slangtorch==1.3.4 "setuptools<80"
```

---

## Usage

```bash
conda activate gaussian_splatting
python rt_gs_gui.py -m ./output/<scene_name>
```

The viewer expects a trained 3DGS scene at `./output/<scene_name>/` (COLMAP format with a `point_cloud/` directory).

### Training a scene (from scratch)

```bash
# 1. Train the base 3DGS scene
python train_scene.py -s <data_dir> -m output/<scene_name>

# 2. Train contrastive CLIP features for segmentation
python train_contrastive_feature.py -s <data_dir> -m output/<scene_name>
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

---

## Project Structure

```
3DGS-RTMaterial/
├── rt_gs_gui.py               # Main viewer (entry point)
├── saga_gui.py                # Original SAGA GUI (segmentation only)
├── optix_integration/         # OptiX / 3DGRT integration module
│   ├── optix_renderer.py      #   BVH build + per-frame ray trace
│   ├── gaussian_adapter.py    #   Wraps SAGA GaussianModel for 3DGRT API
│   ├── ray_generator.py       #   Camera → ray batch conversion
│   ├── material_compositor.py #   Blinn-Phong shading on ray-traced normals
│   └── build_plugin.py        #   Compiles the 3DGRT Slang/CUDA plugin
├── material_sh_edit/          # SH-based material editor utilities
├── 3dgrut/                    # NVIDIA 3DGRT library (submodule)
├── scene/                     # Scene + GaussianModel classes
├── gaussian_renderer/         # 3DGS CUDA rasterizer wrappers
├── submodules/                # diff-gaussian-rasterization, simple-knn, …
├── demo/                      # Screenshots and demo video
├── environment.yml            # Conda environment spec
└── setup.sh                   # One-shot environment setup
```

---

## Known Issues / Limitations

- The 3DGRT OptiX plugin compiles Slang/CUDA kernels on **first run** — this takes 1–3 minutes. Subsequent runs use the cached binary.
- `slangtorch >= 1.3.5` is incompatible with the current kernel sources; pin to `1.3.4`.
- `setuptools >= 80` removes `pkg_resources`; pin to `< 80`.
- Python 3.10 is required (3DGRT was originally 3.11-only; `setup.py` has been patched).
- Material shading is skipped during camera drag for interactive frame rates; full shading resumes when the camera stops.

---

## Acknowledgements

- [SAGA — Segment Any 3D GAussians](https://github.com/Jumpat/SegAnyGAussians)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [NVIDIA 3DGRT / threedgrut](https://github.com/nv-tlabs/3dgrut)
- [CLIP](https://github.com/openai/CLIP)
