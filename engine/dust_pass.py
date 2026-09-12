"""Dust particle pass: tiny alpha-blended motes drifting in the air.

A fixed pool of MAX_DUST random seed positions is uploaded once; each frame
the vertex shader drifts and camera-wraps them (see shaders/dust.vert), and
only the first `count` (from the quality level) are drawn. Rendered into the
HDR target's own framebuffer so the scene's depth buffer occludes motes
behind geometry, with depth writes off and alpha blending on so overlapping
motes accumulate softly. Physically it's a cheap stand-in for airborne
particulate, not a simulation - but it's tunable (count/size/speed/opacity/
colour) exactly as the tool asks.
"""
from __future__ import annotations

import numpy as np
import moderngl
from OpenGL import GL

from .shader import load_program
from .gl_math import mat4_to_array

MAX_DUST = 9000       # matches settings.DUST_PARAMS level-3 count
WRAP_BOX_HALF = 18.0  # motes live in a +-18 unit cube centred on the camera


class DustPass:
    def __init__(self, ctx: moderngl.Context, hdr_target) -> None:
        self.ctx = ctx
        self.hdr = hdr_target  # HDRTarget: use its .fbo (color + scene depth) each frame
        self.program = load_program(ctx, "dust.vert", "dust.frag")

        rng = np.random.default_rng(20240607)
        seeds = rng.uniform(0.0, 2.0 * WRAP_BOX_HALF, size=(MAX_DUST, 3)).astype(np.float32)
        self.vbo = ctx.buffer(seeds.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.vbo, "3f", "in_seed")])
        self.program["u_box"].value = WRAP_BOX_HALF

    def render(self, view_proj, cam_pos, time_s: float, size: tuple[int, int], params: dict) -> None:
        count = int(params["count"])
        if count <= 0:
            return
        ctx = self.ctx
        self.hdr.fbo.use()
        ctx.viewport = (0, 0, *size)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.enable(ctx.BLEND)
        ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA
        # Depth test against the scene (LEQUAL) but never write depth - motes
        # must not occlude each other or later passes. moderngl has no depth
        # write-mask property, so drive it via raw GL and restore after.
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)  # let the vertex shader set gl_PointSize

        self.program["u_view_proj"].write(mat4_to_array(view_proj).tobytes())
        self.program["u_cam_pos"].value = (cam_pos.x, cam_pos.y, cam_pos.z)
        self.program["u_time"].value = float(time_s)
        self.program["u_speed"].value = float(params["speed"])
        self.program["u_size"].value = float(params["size"])
        self.program["u_screen"].value = (float(size[0]), float(size[1]))
        self.program["u_color"].value = tuple(params["color"])
        self.program["u_opacity"].value = float(params["opacity"])

        self.vao.render(mode=moderngl.POINTS, vertices=min(count, MAX_DUST))

        GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)
        # Restore moderngl's defaults (depth writes on, LESS) so later passes
        # and the next frame's forward pass behave normally.
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        ctx.disable(ctx.BLEND)
