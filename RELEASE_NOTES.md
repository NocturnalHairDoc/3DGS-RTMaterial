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

See `docs/V3_PBR_LITE.md` for architecture, measurements, paper basis, and the
recommended next stage.

## Verification

- 35 CPU-safe unit/CLI tests.
- GPU-tested with 61,380, 1,215,289, and 1,070,314 Gaussians.
- GUI startup and real 3DGRT arbitrary-ray tracing validated.
- Removes model-content validation from saved project state; compatibility is
  checked with scene name and Gaussian count only.

---

# v2.2-instance-graph

## Cross-view instances

- Groups spatially neighbouring Gaussians into a configurable sparse anchor graph.
- Projects SAM masks with a point-centre z-buffer and casts mask votes to anchors.
- Associates cross-view mask nodes using 3D overlap, visibility, lightweight RGB
  mask context, 32-D SAGA features, and anchor-graph connectivity.
- Resolves instances with unary voting, graph smoothing, and spatial connected cuts.
- Refines individual Gaussians only inside anchors shared by different instances.
- Adds a reproducible real-data V2.1/V2.2 comparison on Mip-NeRF360 bicycle.
- Replaces the unsupported `torch.flatnonzero` call on the real-data path with a
  PyTorch-version-compatible expression.

See `docs/V2.2_INSTANCE_GRAPH.md` for design, measurements, limitations, and the
next update plan.

## Verification

- 29/29 CPU-safe unit tests pass.
- Full real bicycle run: 61,380 Gaussians, 194 views, 1,911 bounded-size anchors.
- Only 3,691 Gaussians (6.0%) in cross-instance boundary anchors required point-level refinement.

---

# v2.1-stable
