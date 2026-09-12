"""Auto-exposure: GPU-only luminance adaptation, no CPU readback.

Two passes, both fullscreen triangles (see fullscreen.py):
1. `luminance.frag` downsamples the HDR frame into a small log2(luminance)
   capture, then `Texture.build_mipmaps()` (confirmed available on ModernGL,
   core GL, no compute shader needed) reduces it the rest of the way.
2. `luminance_adapt.frag` is a 1x1 draw that reads that chain's coarsest
   level (the single texel averaging the whole capture) and blends it with
   the *previous frame's* adapted value, ping-ponged between two 1x1
   textures so this pass never reads and writes the same image in one draw
   (the same feedback-loop hazard volumetric_pass.py's docstring already
   flags for a different pair of textures).

Deliberately not a CPU `Texture.read()` readback + Python-side EMA: unlike
gpu_timing.py's ring-buffer (which exists because ModernGL's GPU timer query
only exposes a blocking accessor - a real, forced constraint), nothing here
requires leaving the GPU at all, so a CPU round trip would only add a
sync-stall risk for no benefit.
"""
from __future__ import annotations

import math

import numpy as np
import moderngl

from .shader import load_program
from .fullscreen import FullscreenPass

LUMINANCE_MAP_UNIT = 13
PREV_ADAPTED_UNIT = 14

MIDDLE_GREY = 0.18  # standard photographic "middle grey" reference luminance


class ExposurePass:
    def __init__(self, ctx: moderngl.Context, hdr_size: tuple[int, int], capture_shift: int) -> None:
        self.ctx = ctx

        self.luminance_program = load_program(ctx, "fullscreen.vert", "luminance.frag")
        self.luminance_program["u_hdr_color"].value = LUMINANCE_MAP_UNIT
        self.luminance_pass = FullscreenPass(ctx, self.luminance_program)

        self.adapt_program = load_program(ctx, "fullscreen.vert", "luminance_adapt.frag")
        self.adapt_program["u_luminance_map"].value = LUMINANCE_MAP_UNIT
        self.adapt_program["u_prev_adapted"].value = PREV_ADAPTED_UNIT
        self.adapt_pass = FullscreenPass(ctx, self.adapt_program)

        init_log = math.log2(MIDDLE_GREY)
        self._adapted = [self._make_1x1(init_log), self._make_1x1(init_log)]
        self._adapted_fbo = [ctx.framebuffer(color_attachments=[t]) for t in self._adapted]
        self._write_index = 0  # the OTHER index holds the last frame's adapted value

        self.luminance_tex: moderngl.Texture | None = None
        self.luminance_fbo: moderngl.Framebuffer | None = None
        self.capture_size = (0, 0)
        self.capture_shift = -1
        self.set_capture_size(hdr_size, capture_shift)

    def _make_1x1(self, value: float) -> moderngl.Texture:
        data = np.array([value], dtype=np.float32).tobytes()
        tex = self.ctx.texture((1, 1), 1, data, dtype="f4")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return tex

    def set_capture_size(self, hdr_size: tuple[int, int], capture_shift: int) -> None:
        """No-op if neither the HDR size nor the quality-level-driven shift
        actually changed - mirrors ShadowPass.set_resolution/MSAATarget.
        capture_shift <= 0 disables auto-exposure entirely (see render())."""
        size = (max(hdr_size[0] >> max(capture_shift, 0), 1), max(hdr_size[1] >> max(capture_shift, 0), 1))
        if size == self.capture_size and capture_shift == self.capture_shift:
            return
        self._release_capture()
        self.capture_shift = capture_shift
        self.capture_size = size
        if capture_shift <= 0:
            return
        self.luminance_tex = self.ctx.texture(size, 1, dtype="f2")
        self.luminance_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.luminance_tex.repeat_x = False
        self.luminance_tex.repeat_y = False
        self.luminance_fbo = self.ctx.framebuffer(color_attachments=[self.luminance_tex])

    def _release_capture(self) -> None:
        if self.luminance_fbo is not None:
            self.luminance_fbo.release()
            self.luminance_tex.release()
            self.luminance_fbo = None
            self.luminance_tex = None

    @property
    def enabled(self) -> bool:
        return self.luminance_fbo is not None

    @property
    def adapted_luminance(self) -> moderngl.Texture:
        """The most recently written adapted-luminance texture (this frame's
        result once render() has run, otherwise last frame's)."""
        return self._adapted[self._write_index]

    def render(self, hdr_color: moderngl.Texture, dt_s: float, tau: float = 0.6) -> None:
        if not self.enabled:
            return
        ctx = self.ctx
        prev_index = self._write_index
        write_index = 1 - prev_index

        ctx.disable(ctx.DEPTH_TEST)

        self.luminance_fbo.use()
        ctx.viewport = (0, 0, *self.capture_size)
        hdr_color.use(location=LUMINANCE_MAP_UNIT)
        self.luminance_pass.draw()
        self.luminance_tex.build_mipmaps()

        self._adapted_fbo[write_index].use()
        ctx.viewport = (0, 0, 1, 1)
        self.luminance_tex.use(location=LUMINANCE_MAP_UNIT)
        self._adapted[prev_index].use(location=PREV_ADAPTED_UNIT)
        self.adapt_program["u_dt"].value = dt_s
        self.adapt_program["u_tau"].value = tau
        self.adapt_pass.draw()

        self._write_index = write_index
