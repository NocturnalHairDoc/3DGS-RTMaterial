"""V3 viewer: retained Stylized SH Edit plus manual PBR-lite rendering."""

# Keep command help CPU-safe, matching the V2 entry point.
if __name__ == "__main__":
    import sys as _early_sys
    if any(arg in {"-h", "--help"} for arg in _early_sys.argv[1:]):
        from argparse import ArgumentParser as _EarlyArgumentParser
        parser = _EarlyArgumentParser(description="3DGS-RTMaterial V3 PBR-lite viewer")
        parser.add_argument("-m", "--model_path")
        parser.add_argument("-f", "--feature_iteration", type=int, default=10000)
        parser.add_argument("-s", "--scene_iteration", type=int, default=30000)
        parser.add_argument("--scale", type=float, default=2.0)
        parser.add_argument("--environment", help="optional .hdr/.exr latitude-longitude map")
        parser.print_help()
        raise SystemExit(0)

import math
import os
from argparse import ArgumentParser

import dearpygui.dearpygui as dpg
import torch
import torch.nn.functional as F

from gaussian_renderer import render
from materials.pbr_lite import (
    HDREnvironment,
    PBRLiteCompositor,
    PBRMaterial,
    PBRParameterStore,
    reconstruct_world_positions,
    reflection_directions,
    refraction_directions,
    srgb_to_linear,
    stabilize_gbuffer_normals,
)
from viewer.project_state import load_project_state
from rt_gs_gui import RTGSConfig
from rt_gs_gui_sh_clip import SHMaterialViewer
from scene import FeatureGaussianModel, GaussianModel


class PBRLiteViewer(SHMaterialViewer):
    """Dual-mode material viewer using a 3DGRT G-buffer for PBR-lite."""

    def __init__(self, opt, gs_model, feat_model, scale_gate, environment_path=None):
        self._pbr_store = None
        self._pbr_compositor = None
        self._environment_path = environment_path
        super().__init__(opt, gs_model, feat_model, scale_gate)
        self._pbr_store = PBRParameterStore.from_gaussian_model(self.engine["scene"])
        environment = self._load_environment(environment_path, quiet=True)
        self._pbr_compositor = PBRLiteCompositor(environment)
        print("[V3] PBR-lite ready; Stylized SH Edit retained as a separate mode.")

    def _rebuild_pbr_store(self):
        self._pbr_store = PBRParameterStore.from_gaussian_model(self.engine["scene"])
        self._rehydrate_pbr_assignments()

    def _load_environment(self, path, quiet=False):
        try:
            if path:
                environment = HDREnvironment.load(path, device="cuda")
                self._environment_path = os.path.abspath(os.path.expanduser(path))
                if not quiet:
                    self._set_io_status(f"Loaded HDR environment: {self._environment_path}")
                return environment
        except Exception as exc:
            if not quiet:
                self._set_io_status(f"HDR load failed, using procedural HDR: {exc}", error=True)
        self._environment_path = None
        return HDREnvironment.procedural(device="cuda")

    def fetch_data(self, cam):
        scene = self.engine["scene"]
        before = (scene._mask.data_ptr(), getattr(scene._mask, "_version", 0),
                  int(scene.segment_times))
        result = super().fetch_data(cam)
        after = (scene._mask.data_ptr(), getattr(scene._mask, "_version", 0),
                 int(scene.segment_times))
        if before != after:
            self._rebuild_pbr_store()
            self._dirty = True
        return result

    def _restore_history_state(self, state):
        super()._restore_history_state(state)
        if state is not None and self._pbr_store is not None:
            self._rebuild_pbr_store()

    def _load_environment_from_ui(self):
        path = str(dpg.get_value("_pbr_environment_path") or "").strip()
        environment = self._load_environment(path)
        self._pbr_compositor.environment = environment
        self._dirty = True

    def _pbr_material_from_ui(self):
        return PBRMaterial(
            tuple(float(value) for value in dpg.get_value("_pbr_albedo")[:3]),
            float(dpg.get_value("_pbr_roughness")),
            float(dpg.get_value("_pbr_metallic")),
            float(dpg.get_value("_pbr_opacity")),
            float(dpg.get_value("_pbr_ior")),
        ).validated()

    def _set_pbr_ui(self, material):
        material = material.validated()
        values = {
            "_pbr_albedo": list(material.albedo),
            "_pbr_roughness": material.roughness,
            "_pbr_metallic": material.metallic,
            "_pbr_opacity": material.opacity,
            "_pbr_ior": material.ior,
        }
        for tag, value in values.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)

    def _apply_pbr_assignment(self):
        segment_id = self._parse_segment_select_value(dpg.get_value("_material_segment_select"))
        if segment_id < 1 or self._pbr_store is None:
            self._set_io_status("Select a segment before applying PBR parameters.", error=True)
            return
        material = self._pbr_material_from_ui()
        count = self._pbr_store.apply_to_segment(
            segment_id, self.engine["scene"]._mask, material)
        assignment = self.material_assignments.setdefault(
            segment_id, {"type": "Default", "name": f"Material_{segment_id}"})
        assignment["pbr"] = {
            "albedo": list(material.albedo), "roughness": material.roughness,
            "metallic": material.metallic, "opacity": material.opacity, "ior": material.ior,
        }
        self._dirty = True
        self._observe_history()
        self._set_io_status(f"Applied PBR-lite to segment {segment_id}: {count} Gaussians")

    def _on_material_segment_select(self, value):
        super()._on_material_segment_select(value)
        segment_id = self._parse_segment_select_value(value)
        pbr = self.material_assignments.get(segment_id, {}).get("pbr")
        if pbr:
            self._set_pbr_ui(PBRMaterial(**pbr))

    def _rehydrate_pbr_assignments(self):
        if self._pbr_store is None:
            return
        for segment_id, assignment in self.material_assignments.items():
            if assignment.get("pbr"):
                self._pbr_store.apply_to_segment(
                    int(segment_id), self.engine["scene"]._mask,
                    PBRMaterial(**assignment["pbr"]),
                )

    def _load_project(self):
        metadata = None
        try:
            path = self._scoped_path(dpg.get_value("_project_state_path").strip(),
                                     "segmentation_res")
            _, metadata = load_project_state(path)
            confirming = self._pending_state_load == os.path.abspath(path)
        except Exception:
            confirming = False
        super()._load_project()
        # The first click previews metadata; the confirmed call replaces the assignments.
        if confirming and self._pending_state_load is None and metadata is not None:
            self._rebuild_pbr_store()
            if dpg.does_item_exist("_pbr_exposure"):
                dpg.set_value("_pbr_exposure", float(metadata.get("pbr_exposure", 0.0)))
            environment_path = metadata.get("environment_path")
            if environment_path:
                self._pbr_compositor.environment = self._load_environment(environment_path)

    def _state_metadata(self):
        metadata = super()._state_metadata()
        metadata["render_pipeline"] = "V3 PBR-lite"
        metadata["environment_path"] = self._environment_path
        metadata["pbr_exposure"] = (float(dpg.get_value("_pbr_exposure"))
                                    if dpg.does_item_exist("_pbr_exposure") else 0.0)
        return metadata

    def _primary_world_rays(self, cam):
        batch = self._optix_renderer._ray_batch(cam)
        transform = batch.T_to_world[0]
        rotation, translation = transform[:3, :3], transform[:3, 3]
        directions = F.normalize(batch.rays_dir[0] @ rotation.T, dim=-1)
        origins = batch.rays_ori[0] @ rotation.T + translation
        return origins, directions

    def _light(self, device):
        az = math.radians(float(dpg.get_value("_pbr_light_az")))
        el = math.radians(float(dpg.get_value("_pbr_light_el")))
        direction = torch.tensor(
            [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)],
            device=device, dtype=torch.float32,
        )
        intensity = float(dpg.get_value("_pbr_light_intensity"))
        return F.normalize(direction, dim=0), torch.full((3,), intensity, device=device)

    def _property_maps(self, cam, pipe, hide_mask):
        black = torch.zeros(3, device="cuda")
        ones = torch.ones_like(self._pbr_store.albedo)
        coverage = render(
            cam, self.engine["scene"], pipe, black,
            override_color=ones, filtered_mask=hide_mask,
        )["render"].permute(1, 2, 0)[..., 0:1].clamp(0, 1)
        divisor = coverage.clamp_min(1e-5)
        albedo = render(
            cam, self.engine["scene"], pipe, black,
            override_color=self._pbr_store.albedo, filtered_mask=hide_mask,
        )["render"].permute(1, 2, 0) / divisor
        packed = render(
            cam, self.engine["scene"], pipe, black,
            override_color=self._pbr_store.packed_properties(), filtered_mask=hide_mask,
        )["render"].permute(1, 2, 0) / divisor
        encoded_ior = ((self._pbr_store.ior - 1.0) / 1.5).expand(-1, 3)
        ior = render(
            cam, self.engine["scene"], pipe, black,
            override_color=encoded_ior, filtered_mask=hide_mask,
        )["render"].permute(1, 2, 0)[..., 0:1] / divisor
        valid = coverage > 1e-5
        albedo = torch.where(valid, albedo.clamp(0, 1), torch.zeros_like(albedo))
        roughness = torch.where(valid, packed[..., 0:1].clamp(0.04, 1),
                                torch.full_like(coverage, 0.5))
        metallic = torch.where(valid, packed[..., 1:2].clamp(0, 1),
                               torch.zeros_like(coverage))
        opacity = torch.where(valid, packed[..., 2:3].clamp(0, 1),
                              torch.ones_like(coverage))
        ior = torch.where(valid, 1.0 + 1.5 * ior.clamp(0, 1),
                          torch.full_like(coverage, 1.5))
        return albedo, roughness, metallic, opacity, ior

    def _render_pbr(self, cam, pipe):
        if self._optix_renderer is None or not self._optix_renderer.available:
            self._backend_error = "PBR-lite requires the 3DGRT G-buffer"
            return super()._render_raytracing(cam, pipe)
        primary = self._optix_renderer.render(
            cam, segment_mask=self._visible_gaussian_mask())
        if primary is None:
            return super()._render_raytracing(cam, pipe)
        if self.moving or self.moving_middle:
            return primary["rgb"].permute(2, 0, 1)

        hide_mask = self._get_hide_mask()
        albedo, roughness, metallic, opacity, ior = self._property_maps(cam, pipe, hide_mask)
        normals = primary["normals"]
        depth = primary["depth"]
        ray_origins, ray_directions = self._primary_world_rays(cam)
        view_dirs = -ray_directions
        positions = reconstruct_world_positions(ray_origins, ray_directions, depth)
        normals = stabilize_gbuffer_normals(normals, positions, depth, view_dirs)
        light_dir, light_radiance = self._light(albedo.device)
        hit = (depth > 0).unsqueeze(-1)
        epsilon = depth[depth > 0].median() * 1e-2 if torch.any(depth > 0) else depth.new_tensor(1e-2)

        shadow_visibility = torch.ones_like(roughness)
        if dpg.get_value("_pbr_shadows"):
            shadow_origins = positions + normals * epsilon
            shadow_dirs = light_dir.view(1, 1, 3).expand_as(shadow_origins)
            shadow = self._optix_renderer.trace_world_rays(shadow_origins, shadow_dirs)
            if shadow is not None:
                shadow_visibility = (1 - shadow["opacity"]).clamp(0.03, 1)
                shadow_visibility = torch.where(hit, shadow_visibility, torch.ones_like(shadow_visibility))

        reflected_linear = refracted_linear = None
        if dpg.get_value("_pbr_secondary_rays"):
            reflected_dirs = reflection_directions(ray_directions, normals, roughness)
            reflected = self._optix_renderer.trace_world_rays(
                positions + normals * epsilon, reflected_dirs)
            if reflected is not None:
                env = self._pbr_compositor.environment.sample(reflected_dirs, roughness)
                reflected_linear = torch.where(
                    reflected["opacity"] > 1e-3,
                    srgb_to_linear(reflected["rgb"]), env)
            refracted_dirs = refraction_directions(ray_directions, normals, ior)
            refracted = self._optix_renderer.trace_world_rays(
                positions - normals * epsilon, refracted_dirs)
            if refracted is not None:
                env = self._pbr_compositor.environment.sample(refracted_dirs, roughness)
                refracted_linear = torch.where(
                    refracted["opacity"] > 1e-3,
                    srgb_to_linear(refracted["rgb"]), env)

        self._pbr_compositor.exposure = float(dpg.get_value("_pbr_exposure"))
        shaded = self._pbr_compositor.shade(
            srgb_to_linear(albedo), roughness, metallic, opacity, normals, depth,
            view_dirs, light_dir, light_radiance, shadow_visibility,
            primary_rgb=primary["rgb"], reflected_linear=reflected_linear,
            refracted_linear=refracted_linear, gbuffer_opacity=primary["opacity"],
        )
        return shaded.permute(2, 0, 1)

    @torch.no_grad()
    def _render_raytracing(self, cam, pipe):
        mode = (dpg.get_value("_rt_submode") if dpg.does_item_exist("_rt_submode")
                else "Stylized (SH Edit)")
        if mode == "PBR-lite":
            return self._render_pbr(cam, pipe)
        if mode == "Original":
            if self._optix_renderer is not None and self._optix_renderer.available:
                output = self._optix_renderer.render(cam, segment_mask=self._visible_gaussian_mask())
                if output is not None:
                    return output["rgb"].permute(2, 0, 1)
            return render(cam, self.engine["scene"], pipe, self.bg_color,
                          filtered_mask=self._get_hide_mask())["render"]
        if mode == "Compare Stylized/PBR":
            stylized = super()._render_raytracing(cam, pipe)
            pbr = self._render_pbr(cam, pipe)
            result = stylized.clone()
            midpoint = result.shape[2] // 2
            result[:, :, midpoint:] = pbr[:, :, midpoint:]
            result[:, :, midpoint - 1:midpoint + 1] = 1
            return result
        # The inherited renderer treats every non-Original/non-Compare value as
        # its existing edited path, which is now explicitly named Stylized.
        return super()._render_raytracing(cam, pipe)

    def _register_dpg(self):
        super()._register_dpg()
        if dpg.does_item_exist("_rt_submode"):
            dpg.configure_item(
                "_rt_submode",
                items=["Stylized (SH Edit)", "PBR-lite", "Original", "Compare Stylized/PBR"],
                default_value="Stylized (SH Edit)",
            )
        with dpg.collapsing_header(
            parent="_rt_group", label="V3 PBR-lite", default_open=True,
            before="_interactive_rt_header", tag="_pbr_header",
        ):
            dpg.add_text("Manual per-segment parameters; dense per-Gaussian fields.",
                         color=(170, 190, 210), wrap=330)
            dpg.add_input_floatx(tag="_pbr_albedo", label="Albedo", size=3,
                                 default_value=[0.8, 0.8, 0.8], min_value=0, max_value=1)
            dpg.add_slider_float(tag="_pbr_roughness", label="Roughness", default_value=0.5,
                                 min_value=0.04, max_value=1.0)
            dpg.add_slider_float(tag="_pbr_metallic", label="Metallic", default_value=0.0,
                                 min_value=0.0, max_value=1.0)
            dpg.add_slider_float(tag="_pbr_opacity", label="Opacity", default_value=1.0,
                                 min_value=0.0, max_value=1.0)
            dpg.add_slider_float(tag="_pbr_ior", label="IOR", default_value=1.5,
                                 min_value=1.0, max_value=2.5)
            dpg.add_button(label="Apply PBR to selected segment", width=-1,
                           callback=lambda: self._apply_pbr_assignment())
            dpg.add_separator()
            dpg.add_input_text(tag="_pbr_environment_path", label="HDR map",
                               default_value=self._environment_path or "")
            dpg.add_button(label="Load HDR environment", width=-1,
                           callback=lambda: self._load_environment_from_ui())
            dpg.add_slider_float(tag="_pbr_exposure", label="Exposure", default_value=0,
                                 min_value=-4, max_value=4,
                                 callback=lambda: setattr(self, "_dirty", True))
            dpg.add_slider_float(tag="_pbr_light_az", label="Light azimuth", default_value=45,
                                 min_value=0, max_value=360,
                                 callback=lambda: setattr(self, "_dirty", True))
            dpg.add_slider_float(tag="_pbr_light_el", label="Light elevation", default_value=55,
                                 min_value=-10, max_value=90,
                                 callback=lambda: setattr(self, "_dirty", True))
            dpg.add_slider_float(tag="_pbr_light_intensity", label="Light intensity",
                                 default_value=1.5, min_value=0, max_value=12,
                                 callback=lambda: setattr(self, "_dirty", True))
            dpg.add_checkbox(tag="_pbr_shadows", label="Shadow rays + visibility",
                             default_value=True, callback=lambda: setattr(self, "_dirty", True))
            dpg.add_checkbox(tag="_pbr_secondary_rays", label="Reflection/refraction rays",
                             default_value=True, callback=lambda: setattr(self, "_dirty", True))


def main():
    parser = ArgumentParser(description="3DGS-RTMaterial V3 PBR-lite viewer")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-f", "--feature_iteration", type=int, default=10000)
    parser.add_argument("-s", "--scene_iteration", type=int, default=30000)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--environment", default=None)
    args = parser.parse_args()

    opt = RTGSConfig()
    opt.r = args.scale
    opt.window_width, opt.window_height = int(2160 / opt.r), int(1200 / opt.r)
    opt.width, opt.height = opt.window_width, opt.window_height
    opt.control_width, opt.control_height = int(350 * (2 / opt.r)), int(700 * (2 / opt.r))
    opt.font_size = min(28, max(16, int(18 * (2 / opt.r))))
    opt.MODEL_PATH = args.model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration
    for label, path in (("Scene PLY", opt.SCENE_PCD_PATH),
                        ("Feature PLY", opt.FEATURE_PCD_PATH),
                        ("Scale gate", opt.SCALE_GATE_PATH)):
        if not os.path.isfile(path):
            parser.error(f"{label} not found: {path}")
    scene = GaussianModel(opt.sh_degree)
    feature = FeatureGaussianModel(opt.FEATURE_DIM)
    gate = torch.nn.Sequential(torch.nn.Linear(1, opt.FEATURE_DIM), torch.nn.Sigmoid()).cuda()
    PBRLiteViewer(opt, scene, feature, gate, environment_path=args.environment).render()


if __name__ == "__main__":
    main()
