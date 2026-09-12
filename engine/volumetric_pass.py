"""Screen-space volumetric light shafts ("god rays") for the directional light.

Raymarches from the camera to each pixel's actual surface depth (the
forward pass's own HDR depth buffer), accumulating in-scattered light only
where the shadow map says that point along the ray is lit - see
shaders/volumetric.frag for the per-pixel math. Additively blended straight
into the HDR color target, before tonemapping, so it's compressed by the
same ACES curve as everything else rather than needing its own exposure
handling.
"""
from __future__ import annotations

import moderngl

from .shader import load_program
from .fullscreen import FullscreenPass
from .frame_ubo import FrameUBO
from .gl_math import mat4_to_array

SCENE_DEPTH_UNIT = 10
SHADOW_MAP_UNIT = 11


class VolumetricPass:
    def __init__(self, ctx: moderngl.Context, hdr_color_texture: moderngl.Texture) -> None:
        self.ctx = ctx
        self.program = load_program(ctx, "fullscreen.vert", "volumetric.frag")
        FrameUBO.bind_to_program(self.program)
        self.program["u_depth_map"].value = SCENE_DEPTH_UNIT
        self.program["u_shadow_map"].value = SHADOW_MAP_UNIT
        # Recalibrated down from an original 0.02 (tuned against a small
        # ~10-12 world-unit demo scene) after the Phase 3 scene expansion
        # added real geometry (walls) ~15-20 units out: this raymarch
        # integrates lit-ness over the *entire* camera-to-surface segment
        # with no notion of "how close to an occluder's edge", so any long,
        # mostly-unoccluded ray (a distant wall with clear air in front of
        # it, not just an empty sky) accumulates a large lit_path_length
        # regardless of whether there's a nearby shadow to actually catch
        # light on - found by reading back raw HDR values and bisecting
        # which pass contributed the wall's unexpectedly high brightness.
        # 0.003 keeps that same-shaped effect visible near the shadowed
        # prop cluster without washing out anything merely far away.
        self.program["u_density"].value = 0.003
        self.fullscreen = FullscreenPass(ctx, self.program)

        # Color-only view of the HDR target's color attachment: this pass's
        # depth input (`u_depth_map`, the SAME texture the forward pass just
        # wrote) must not also be attached as this draw's own depth buffer -
        # reading and having-attached the same image in one draw is a GL
        # feedback loop. Using a depth-less FBO wrapper here sidesteps that
        # entirely rather than relying on depth test/write being disabled.
        self.color_only_fbo = ctx.framebuffer(color_attachments=[hdr_color_texture])

    def rebind_target(self, hdr_color_texture: moderngl.Texture) -> None:
        """Call after HDRTarget.resize() swaps in a new color texture."""
        self.color_only_fbo.release()
        self.color_only_fbo = self.ctx.framebuffer(color_attachments=[hdr_color_texture])

    def render(self, scene_depth: moderngl.Texture, shadow_map: moderngl.Texture,
               inv_view_proj, steps: int, size: tuple[int, int]) -> None:
        if steps <= 0:
            return
        ctx = self.ctx
        self.color_only_fbo.use()
        ctx.viewport = (0, 0, *size)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.enable(ctx.BLEND)
        ctx.blend_func = ctx.ONE, ctx.ONE  # additive: only ever adds light, never darkens

        scene_depth.use(location=SCENE_DEPTH_UNIT)
        shadow_map.use(location=SHADOW_MAP_UNIT)
        self.program["u_inv_view_proj"].write(mat4_to_array(inv_view_proj).tobytes())
        self.program["u_steps"].value = steps
        self.fullscreen.draw()

        ctx.disable(ctx.BLEND)
