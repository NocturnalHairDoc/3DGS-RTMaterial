# V3 PBR-lite design and validation

## Scope

V3 adds a manually controlled physically based rendering layer without removing
the existing SH editor. `Stylized (SH Edit)` remains the V2.2 rendering path;
`PBR-lite` is a separate mode. It is intentionally a forward renderer, not an
inverse-rendering claim: the user supplies material values and the trained SH
appearance remains available as the original/stylized reference.

## Data and rendering flow

1. `PBRParameterStore` allocates dense `N×3` albedo and `N×1` roughness,
   metallic, opacity, and IOR tensors. The public API can edit arbitrary Gaussian
   indices; the GUI writes the Gaussians belonging to the selected segment.
2. Gaussian rasterization converts those dense fields to screen-space property
   maps. Each field is divided by a matching coverage pass so values do not
   darken at partially covered pixels. 3DGRT returns primary RGB, normal, depth,
   and opacity buffers.
3. World position is reconstructed from the primary ray and depth. A guarded
   depth-tangent estimate is blended with the traced Gaussian normal away from
   depth discontinuities.
4. The direct term uses Cook–Torrance with a GGX distribution, Smith–Schlick
   masking-shadowing, and Schlick Fresnel. The indirect term samples a floating-
   point latitude-longitude environment; `.hdr` and `.exr` files are accepted,
   with a deterministic procedural HDR fallback.
5. Arbitrary world rays are submitted through the same 3DGRT BVH. One ray toward
   the light estimates visibility. Reflection and Snell refraction rays use
   traced RGB on a hit and the HDR environment on a miss. This follows the
   hybrid secondary-ray design demonstrated by 3DGUT/3DGRUT, rather than claiming
   a new path tracer.
6. Linear lighting is exposure-adjusted, ACES-mapped, and encoded to sRGB.

## Controls and persistence

Launch:

```bash
conda activate gaussian_splatting_v2
python rt_gs_gui_v3.py -m /absolute/model/path --scale 1.5

# Universal import and automatic initial segmentation
python rt_gs_gui_v3.py --one-click -m /absolute/model/or/scene.ply
```

Choose `Ray-Tracing → PBR-lite`, select a segment, set albedo, roughness,
metallic, opacity, and IOR, then apply. HDR path, exposure, light direction and
intensity, shadows, and secondary rays are explicit controls. Version-3 project
state stores segment PBR assignments, HDR path, and exposure; older state files
are migrated in memory. Dense fields are rebuilt from those assignments after
load, rollback, clear, undo, and redo. Refraction reads the rasterized IOR field
rather than the currently selected UI value.

The one-click resolver accepts a model directory, `point_cloud` directory, or
direct trained PLY and selects the latest usable iteration. SAGA assets retain
their learned feature/gate pair. Plain 3DGS scenes use deterministic
geometry/appearance proxy features and KMeans segmentation. These proxy
clusters may not correspond to semantic objects.

`Project & Export → Render pipeline` freezes the chosen PBR pipeline together
with dense material tensors, environment, exposure, lighting, shadows, and
secondary-ray toggles. PNG, MP4, and PNG-sequence workers therefore use the
same PBR/G-buffer/compositor path as the settled interactive preview.

## Real-data comparison

The benchmark uses the same trained Gaussian model and camera for all five
columns. The V2.2 controls are unedited Original and its all-scene Metal SH edit;
V3 renders three manual material rounds with HDR, GGX, visibility, reflection,
and refraction enabled. Uniform all-scene materials isolate renderer behavior;
interactive use remains segment-specific.

| Scene | Gaussians | Resolution | Tracing/compositor FPS | Minimum finite fraction |
|---|---:|---:|---:|---:|
| bicycle smoke scene | 61,380 | 618×411 | 11.8 | 1.0 |
| Mip-NeRF360 bonsai | 1,215,289 | 390×260 | 10.4 | 1.0 |
| Mip-NeRF360 counter | 1,070,314 | 389×260 | 6.9 | 1.0 |

Per-scene and aggregate measurements are committed as JSON files under
`comparisons/`.
Timing is a synchronized single-machine smoke measurement of primary, shadow,
and secondary tracing plus composition. It excludes property-map rasterization,
GUI work, and image encoding, so it is not an end-to-end viewer frame rate or a
cross-hardware benchmark.
PBR-lite still-image and turntable export were validated on both a learned-SAGA
scene and a plain degree-0 SH scene imported from another 3DGS implementation.

## Evidence and design basis

- [3D Gaussian Ray Tracing](https://arxiv.org/abs/2407.07090) supplies the
  explicit Gaussian ray-intersection/G-buffer basis.
- [3DGUT](https://arxiv.org/abs/2412.12507) and the
  [official 3DGRUT implementation](https://github.com/nv-tlabs/3DGRUT) motivate
  hybrid arbitrary secondary rays for shadows, reflection, and refraction.
- The [Khronos glTF PBR overview](https://www.khronos.org/gltf/pbr) provides the
  conventional metallic-roughness parameter interpretation.

## Known limitations

The current images show strong ellipse/splat contour responses in PBR modes,
especially on million-Gaussian indoor scenes. These originate in the local
3DGRT intersection normals and cannot be fully repaired with image-space depth
derivatives. Material values are also user-authored, so baked illumination can
remain in the albedo initialization.
