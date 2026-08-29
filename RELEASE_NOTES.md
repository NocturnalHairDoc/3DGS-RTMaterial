# v3.2-multiview-selection

## Calibrated photo selection to colored 3D instances

- Adds an interactive selector for choosing one physical object in two or more
  SAM-processed training photographs. Whole-object and fine-parts brushes can
  select multiple proposals, which are unioned once per view before 3D voting.
- Persists portable JSON selection manifests and accepts them in both the V3
  GUI and the standalone SAM-to-Gaussian pipeline.
- Corrects SAM projection to use the CUDA rasterizer's row-vector transform and
  image-axis convention, with visibility-aware fusion and direct preservation
  of selected Gaussian evidence.
- Adds explicit single-object correspondence and requires `min_votes` support
  per Gaussian, preventing correct masks from being rejected while suppressing
  one-view background splat leakage.
- Adds calibrated multi-view result rendering and neutralizes unselected
  Gaussians so every recovered instance is displayed in a distinct color.

---

# v3.1-universal-import-export

## One-click trained-scene workflow

- Accepts model directories, `point_cloud` directories, standard or SAGA PLYs,
  and automatically selects the latest usable scene/feature iterations.
- Preserves learned SAGA segmentation when available and creates deterministic
  geometry/appearance proxy features for plain 3DGS scenes.
- Adds startup HDBSCAN/KMeans/SAM selection, a universal KMeans fallback, robust
  small-scene handling, and camera fitting for plain imported scenes.
- Loads degree-0 through degree-3 SH PLYs and pads lower-order coefficients for
  the degree-3 renderer, enabling compatible outputs from other 3DGS projects.
- Adds frozen PBR-lite PNG, turntable MP4, and PNG-sequence export using the same
  dense maps, OptiX G-buffer, HDR, light, shadow, and secondary-ray state as the
  interactive renderer.

---

# v3.0-pbr-lite

## Manual PBR-lite pipeline

- Keeps the V2.2 SH material editor as the explicit `Stylized (SH Edit)` mode.
- Adds dense per-Gaussian albedo, roughness, metallic, opacity, and IOR fields,
  with segment-level editing and version-3 project-state persistence.
- Normalizes rasterized material fields by Gaussian coverage and uses the stored
  per-Gaussian IOR map for refraction.
- Rebuilds dense PBR fields after project load, rollback, clear, undo, and redo;
  SH/CLIP edits now retain PBR parameters on the same segment.
- Adds HDR environment sampling, ACES output mapping, and Cook–Torrance GGX.
- Uses 3DGRT normal/depth/opacity outputs as a G-buffer, with depth-derived normal
  stabilization, shadow visibility rays, and hybrid reflection/refraction rays.
- Adds three reproducible real-scene V2.2/V3 comparisons: bicycle smoke scene,
  Mip-NeRF360 bonsai, and Mip-NeRF360 counter.
- All tested pixels were finite. Synchronized tracing/compositing measurements
  are reported separately from property-map rasterization and GUI overhead.

See `docs/V3_PBR_LITE.md` for architecture, measurements, paper basis, and
limitations.

## Verification

- 35 CPU-safe unit/CLI tests.
- GPU-tested with 61,380, 1,215,289, and 1,070,314 Gaussians.
- GUI startup and real 3DGRT arbitrary-ray tracing validated.
- Removes model-content validation from saved project state; compatibility is
  checked with scene name and Gaussian count only.

---

# v2.2-instance-graph

## Cross-view instances

- Groups spatially neighboring Gaussians into a configurable sparse anchor graph.
- Projects SAM masks with a point-center z-buffer and casts mask votes to anchors.
- Associates cross-view mask nodes using 3D overlap, visibility, lightweight RGB
  mask context, 32-D SAGA features, and anchor-graph connectivity.
- Resolves instances with unary voting, graph smoothing, and spatial connected cuts.
- Refines individual Gaussians only inside anchors shared by different instances.
- Adds a reproducible real-data V2.1/V2.2 comparison on Mip-NeRF360 bicycle.
- Replaces the unsupported `torch.flatnonzero` call on the real-data path with a
  PyTorch-version-compatible expression.

See `docs/V2.2_INSTANCE_GRAPH.md` for design, measurements, and limitations.

## Verification

- 29/29 CPU-safe unit tests pass.
- Full real bicycle run: 61,380 Gaussians, 194 views, 1,911 bounded-size anchors.
- Only 3,691 Gaussians (6.0%) in cross-instance boundary anchors required point-level refinement.

---

# v2.1-stable
