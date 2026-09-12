"""Orchestrates the per-frame render passes.

Pipeline, in order: shadow pass (directional-light depth-only) -> depth+
normal prepass and SSAO -> forward PBR pass, including inline hybrid
ray-traced reflections (writes linear HDR into render_targets.HDRTarget) ->
volumetric light shafts (additive, into the same HDR target) -> tonemap pass
(ACES + gamma, composites to the default framebuffer). Every pass reads its
quality level from a shared `settings.QualitySettings` (0 = off, 1-3 = an
increasing resolution/sample-count tradeoff) rather than a plain on/off
flag - see settings.py for what each level means concretely.

All GPU work for a frame is wrapped in one non-blocking timer
(gpu_timing.GPUFrameTimer). main.py drives this with one call per frame:
`renderer.render(scene, camera, time_s, dt_s)`.
"""
from __future__ import annotations

import glm
import moderngl

from .frame_ubo import FrameUBO
from .render_targets import HDRTarget, MSAATarget
from .shadow_pass import ShadowPass
from .prepass import PrepassTarget
from .ssao_pass import SSAOPass, AO_TEXTURE_UNIT
from .rt_primitives import RTPrimitivesUBO
from .volumetric_pass import VolumetricPass
from .exposure_pass import ExposurePass
from .fullscreen import FullscreenPass
from .shader import load_program
from .scene import Scene
from .camera import Camera
from .settings import QualitySettings
from .material_textures import MaterialTextures
from .gl_math import mat4_to_array, mat3_to_array, normal_matrix
from .gpu_timing import GPUFrameTimer

SHADOW_MAP_TEXTURE_UNIT = 1  # unit 0 is reserved for the tonemap pass's HDR source
# Material texture units - distinct from SHADOW_MAP_TEXTURE_UNIT(1)/
# AO_TEXTURE_UNIT(2) and ssao_pass.py's 6/7/8/9, all bound simultaneously
# during the forward pass's per-object draw loop.
ALBEDO_MAP_UNIT = 3
ROUGHNESS_MAP_UNIT = 4
METALLIC_MAP_UNIT = 5
NORMAL_MAP_UNIT = 12


class Renderer:
    def __init__(self, ctx: moderngl.Context, framebuffer_size: tuple[int, int],
                 settings: QualitySettings | None = None) -> None:
        self.ctx = ctx
        self.settings = settings or QualitySettings()
        self.size = framebuffer_size

        self.frame_ubo = FrameUBO(ctx)
        self.forward_program = load_program(ctx, "forward.vert", "forward.frag")
        FrameUBO.bind_to_program(self.forward_program)
        self.forward_program["u_shadow_map"].value = SHADOW_MAP_TEXTURE_UNIT
        self.forward_program["u_ao_map"].value = AO_TEXTURE_UNIT
        self.forward_program["u_albedo_map"].value = ALBEDO_MAP_UNIT
        self.forward_program["u_roughness_map"].value = ROUGHNESS_MAP_UNIT
        self.forward_program["u_metallic_map"].value = METALLIC_MAP_UNIT
        self.forward_program["u_normal_map"].value = NORMAL_MAP_UNIT
        self.material_textures = MaterialTextures(ctx)
        self.rt_primitives = RTPrimitivesUBO(ctx)
        RTPrimitivesUBO.bind_to_program(self.forward_program)

        self.shadow_pass = ShadowPass(ctx)
        self.prepass = PrepassTarget(ctx, framebuffer_size)
        self.ssao_pass = SSAOPass(ctx, framebuffer_size)

        self.tonemap_program = load_program(ctx, "fullscreen.vert", "tonemap.frag")
        self.tonemap_program["u_hdr_color"].value = 0
        self.tonemap_program["u_adapted_luminance"].value = 1
        self.tonemap_program["u_auto_exposure_key"].value = 0.18
        self.tonemap_pass = FullscreenPass(ctx, self.tonemap_program)

        self.hdr = HDRTarget(ctx, framebuffer_size)
        self.msaa = MSAATarget(ctx, framebuffer_size, self.settings.msaa_samples)
        self.volumetric_pass = VolumetricPass(ctx, self.hdr.color)
        self.exposure_pass = ExposurePass(ctx, framebuffer_size, self.settings.autoexposure_capture_shift)
        self.gpu_timer = GPUFrameTimer(ctx)

        self._apply_settings()

    def resize(self, size: tuple[int, int]) -> None:
        self.hdr.resize(size)
        self.msaa.resize(size)
        self.prepass.resize(size)
        self.ssao_pass.resize(size)
        self.volumetric_pass.rebind_target(self.hdr.color)
        self.size = size

    def _apply_settings(self) -> None:
        """Push the current QualitySettings into every sub-pass. Each setter
        is a no-op if the relevant value hasn't actually changed (e.g.
        ShadowPass.set_resolution only reallocates on a genuine change), so
        calling this every frame is cheap and keeps the settings UI's
        sliders live without any extra "did it change" bookkeeping here."""
        s = self.settings
        s.clamp()
        self.shadow_pass.set_resolution(s.shadow_map_size)
        self.ssao_pass.set_sample_count(s.ssao_samples)
        self.forward_program["u_shadow_blocker_radius"].value = max(s.shadow_blocker_search_radius, 1)
        self.forward_program["u_shadow_max_radius"].value = max(s.shadow_max_penumbra_radius, 1)
        self.forward_program["u_rt_samples"].value = s.rt_samples
        self.forward_program["u_fog_density"].value = s.fog_density
        self.msaa.set_samples(s.msaa_samples)
        self.exposure_pass.set_capture_size(self.hdr.size, s.autoexposure_capture_shift)

    def render(self, scene: Scene, camera: Camera, time_s: float, dt_s: float) -> None:
        ctx = self.ctx
        self._apply_settings()
        s = self.settings

        view = camera.view_matrix()
        proj = camera.projection_matrix()
        view_proj = proj * view
        dl = scene.directional_light
        pl = scene.point_lights[0]
        bounds_min, bounds_max = scene.world_bounds()
        shadow_frustum = self.shadow_pass.fit_frustum(dl.direction, bounds_min, bounds_max)
        light_view_proj = shadow_frustum.view_proj
        # Per-frame (scene bounds can change frame to frame), unlike the
        # blocker/penumbra radii above which are quality-level constants -
        # PCSS needs these to convert shadow-map device depth back to a
        # world-space distance for its penumbra estimate (forward.frag).
        self.forward_program["u_shadow_near"].value = shadow_frustum.near
        self.forward_program["u_shadow_depth_range"].value = shadow_frustum.far - shadow_frustum.near
        al = scene.ambient_light
        self.frame_ubo.write(
            view=view, proj=proj, view_proj=view_proj, light_view_proj=light_view_proj,
            cam_pos=camera.position,
            dir_light_dir=glm.vec3(*dl.direction), dir_light_color=dl.color,
            dir_light_intensity=dl.intensity, dir_light_softness=dl.softness,
            point_light_pos=glm.vec3(*pl.position), point_light_range=pl.range,
            point_light_color=pl.color, point_light_intensity=pl.intensity,
            ambient_sky_color=al.sky_color, ambient_ground_color=al.ground_color,
            ambient_intensity=al.intensity,
            time_s=time_s, dt_s=dt_s, screen_w=self.size[0], screen_h=self.size[1],
            ao_enabled=s.ssao_level > 0, shadows_enabled=s.shadow_level > 0,
            rt_enabled=s.rt_level > 0,
        )

        # Everything from here down is GPU work for this frame, timed as one
        # whole-frame span (see gpu_timing.py for why not one query per pass).
        with self.gpu_timer.begin():
            # --- shadow pass: directional-light depth-only ---
            if s.shadow_level > 0:
                self.shadow_pass.render(scene, light_view_proj)

            # --- prepass + SSAO: only when enabled, so disabling it also saves the GPU time ---
            if s.ssao_level > 0:
                self.prepass.render(scene)
                self.ssao_pass.render(self.prepass, proj, glm.inverse(proj))

            # Only re-uploaded when an RT-eligible object's transform actually
            # changed since the last upload (see RTPrimitivesUBO.update).
            self.rt_primitives.update(scene)

            # --- forward PBR pass: opaque geometry -> linear HDR target ---
            # Draws into the multisample target when MSAA is active, resolved
            # into self.hdr right after - every pass below this point (and
            # the tonemap pass) only ever reads self.hdr, so none of them
            # need to know MSAA exists at all.
            msaa_active = self.msaa.fbo is not None
            forward_target = self.msaa if msaa_active else self.hdr
            forward_target.use()
            ctx.viewport = (0, 0, *self.size)
            ctx.enable(ctx.DEPTH_TEST)
            ctx.clear(0.0, 0.0, 0.0)

            self.shadow_pass.depth.use(location=SHADOW_MAP_TEXTURE_UNIT)
            self.ssao_pass.blurred_ao.use(location=AO_TEXTURE_UNIT)
            prog = self.forward_program
            mat_tex = self.material_textures
            for obj in scene.iter_visible():
                model = obj.transform.matrix()
                mat = obj.material
                prog["u_model"].write(mat4_to_array(model).tobytes())
                prog["u_normal_matrix"].write(mat3_to_array(normal_matrix(model)).tobytes())
                (mat.albedo_map or mat_tex.get_or_create("albedo", mat.albedo)).use(location=ALBEDO_MAP_UNIT)
                (mat.roughness_map or mat_tex.get_or_create("roughness", mat.roughness)).use(location=ROUGHNESS_MAP_UNIT)
                (mat.metallic_map or mat_tex.get_or_create("metallic", mat.metallic)).use(location=METALLIC_MAP_UNIT)
                (mat.normal_map or mat_tex.default_normal).use(location=NORMAL_MAP_UNIT)
                prog["u_reflectivity"].value = mat.reflectivity
                obj.mesh.vertex_array(prog).render()

            if msaa_active:
                self.msaa.resolve_into(self.hdr)

            # --- volumetric light shafts: additive, into the same HDR target ---
            if s.volumetric_level > 0 and s.shadow_level > 0:
                self.volumetric_pass.render(
                    self.hdr.depth, self.shadow_pass.depth, glm.inverse(view_proj),
                    s.volumetric_steps, self.size,
                )

            # --- auto-exposure: reads the finished HDR frame, entirely GPU-side ---
            self.exposure_pass.render(self.hdr.color, dt_s)

            # --- tonemap composite: HDR target -> default framebuffer ---
            ctx.screen.use()
            ctx.viewport = (0, 0, *self.size)
            ctx.disable(ctx.DEPTH_TEST)
            self.hdr.color.use(location=0)
            self.exposure_pass.adapted_luminance.use(location=1)
            self.tonemap_program["u_exposure"].value = s.exposure
            self.tonemap_program["u_use_auto_exposure"].value = self.exposure_pass.enabled
            self.tonemap_pass.draw()
