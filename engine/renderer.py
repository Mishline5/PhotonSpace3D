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
from .render_targets import HDRTarget
from .shadow_pass import ShadowPass
from .prepass import PrepassTarget
from .ssao_pass import SSAOPass, AO_TEXTURE_UNIT
from .rt_primitives import RTPrimitivesUBO
from .volumetric_pass import VolumetricPass
from .fullscreen import FullscreenPass
from .shader import load_program
from .scene import Scene
from .camera import Camera
from .settings import QualitySettings
from .gl_math import mat4_to_array, mat3_to_array, normal_matrix
from .gpu_timing import GPUFrameTimer

SHADOW_MAP_TEXTURE_UNIT = 1  # unit 0 is reserved for the tonemap pass's HDR source


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
        self.rt_primitives = RTPrimitivesUBO(ctx)
        RTPrimitivesUBO.bind_to_program(self.forward_program)

        self.shadow_pass = ShadowPass(ctx)
        self.prepass = PrepassTarget(ctx, framebuffer_size)
        self.ssao_pass = SSAOPass(ctx, framebuffer_size)

        self.tonemap_program = load_program(ctx, "fullscreen.vert", "tonemap.frag")
        self.tonemap_program["u_hdr_color"].value = 0
        self.tonemap_pass = FullscreenPass(ctx, self.tonemap_program)

        self.hdr = HDRTarget(ctx, framebuffer_size)
        self.volumetric_pass = VolumetricPass(ctx, self.hdr.color)
        self.gpu_timer = GPUFrameTimer(ctx)

        self._apply_settings()

    def resize(self, size: tuple[int, int]) -> None:
        self.hdr.resize(size)
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
        self.forward_program["u_pcf_radius"].value = max(s.shadow_pcf_radius, 1)
        self.forward_program["u_rt_samples"].value = s.rt_samples

    def render(self, scene: Scene, camera: Camera, time_s: float, dt_s: float) -> None:
        ctx = self.ctx
        self._apply_settings()
        s = self.settings

        view = camera.view_matrix()
        proj = camera.projection_matrix()
        view_proj = proj * view
        dl = scene.directional_light
        pl = scene.point_lights[0]
        light_view_proj = self.shadow_pass.light_view_proj(dl.direction)
        self.frame_ubo.write(
            view=view, proj=proj, view_proj=view_proj, light_view_proj=light_view_proj,
            cam_pos=camera.position,
            dir_light_dir=glm.vec3(*dl.direction), dir_light_color=dl.color,
            dir_light_intensity=dl.intensity,
            point_light_pos=glm.vec3(*pl.position), point_light_range=pl.range,
            point_light_color=pl.color, point_light_intensity=pl.intensity,
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
            self.hdr.use()
            ctx.viewport = (0, 0, *self.size)
            ctx.enable(ctx.DEPTH_TEST)
            ctx.clear(0.0, 0.0, 0.0)

            self.shadow_pass.depth.use(location=SHADOW_MAP_TEXTURE_UNIT)
            self.ssao_pass.blurred_ao.use(location=AO_TEXTURE_UNIT)
            prog = self.forward_program
            for obj in scene.iter_visible():
                model = obj.transform.matrix()
                prog["u_model"].write(mat4_to_array(model).tobytes())
                prog["u_normal_matrix"].write(mat3_to_array(normal_matrix(model)).tobytes())
                prog["u_albedo"].value = tuple(obj.material.albedo)
                prog["u_metallic"].value = obj.material.metallic
                prog["u_roughness"].value = obj.material.roughness
                prog["u_reflectivity"].value = obj.material.reflectivity
                obj.mesh.vertex_array(prog).render()

            # --- volumetric light shafts: additive, into the same HDR target ---
            if s.volumetric_level > 0 and s.shadow_level > 0:
                self.volumetric_pass.render(
                    self.hdr.depth, self.shadow_pass.depth, glm.inverse(view_proj),
                    s.volumetric_steps, self.size,
                )

            # --- tonemap composite: HDR target -> default framebuffer ---
            ctx.screen.use()
            ctx.viewport = (0, 0, *self.size)
            ctx.disable(ctx.DEPTH_TEST)
            self.hdr.color.use(location=0)
            self.tonemap_program["u_exposure"].value = s.exposure
            self.tonemap_pass.draw()
