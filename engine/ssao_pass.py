"""Screen-space ambient occlusion: hemisphere-kernel AO from the depth+normal prepass.

Deliberately affects ONLY the ambient/indirect term in forward.frag, never
direct light - see forward.frag's `ao` usage. Two screen-sized targets: a
raw (noisy) AO texture and a small box-blur pass over it, since the raw
kernel's per-pixel random rotation (needed to hide banding with only 24
samples) shows up as visible dither until blurred.

Kernel and noise are generated once at startup with a fixed random seed:
deterministic frame-to-frame (no flicker) and reproducible for benchmark
comparisons, at zero runtime cost since neither is regenerated per frame.
"""
from __future__ import annotations

import numpy as np
import moderngl

from .shader import load_program
from .fullscreen import FullscreenPass
from .gl_math import mat4_to_array

KERNEL_SIZE = 24
NOISE_DIM = 4
RANDOM_SEED = 1234

AO_TEXTURE_UNIT = 2
NOISE_TEXTURE_UNIT = 6
PREPASS_NORMAL_UNIT = 7
PREPASS_DEPTH_UNIT = 8
RAW_AO_UNIT = 9


def _generate_kernel(rng: np.random.Generator) -> np.ndarray:
    kernel = np.zeros((KERNEL_SIZE, 3), dtype=np.float32)
    for i in range(KERNEL_SIZE):
        while True:
            v = rng.uniform(-1.0, 1.0, size=3)
            v[2] = abs(v[2])  # hemisphere oriented around +Z (the surface normal in TBN space)
            if float(np.dot(v, v)) <= 1.0:
                break
        v = v / (np.linalg.norm(v) + 1e-8)
        scale = 0.1 + 0.9 * (i / KERNEL_SIZE) ** 2  # cluster samples closer to the origin
        kernel[i] = v * scale
    return kernel


def _generate_noise(rng: np.random.Generator) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, size=NOISE_DIM * NOISE_DIM)
    noise = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    return noise.astype(np.float32)


class SSAOPass:
    def __init__(self, ctx: moderngl.Context, size: tuple[int, int]) -> None:
        self.ctx = ctx
        rng = np.random.default_rng(seed=RANDOM_SEED)

        self.program = load_program(ctx, "fullscreen.vert", "ssao.frag")
        self.program["u_kernel"].write(_generate_kernel(rng).tobytes())
        self.program["u_sample_count"].value = KERNEL_SIZE
        self.program["u_radius"].value = 0.5
        self.program["u_bias"].value = 0.02
        self.program["u_strength"].value = 1.0
        self.program["u_noise"].value = NOISE_TEXTURE_UNIT
        self.program["u_normal_map"].value = PREPASS_NORMAL_UNIT
        self.program["u_depth_map"].value = PREPASS_DEPTH_UNIT

        noise = _generate_noise(rng)
        self.noise_texture = ctx.texture((NOISE_DIM, NOISE_DIM), 3, noise.tobytes(), dtype="f4")
        self.noise_texture.repeat_x = True
        self.noise_texture.repeat_y = True
        self.noise_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.blur_program = load_program(ctx, "fullscreen.vert", "ssao_blur.frag")
        self.blur_program["u_ao"].value = RAW_AO_UNIT

        self.fullscreen = FullscreenPass(ctx, self.program)
        self.blur_fullscreen = FullscreenPass(ctx, self.blur_program)

        self.raw_ao: moderngl.Texture | None = None
        self.raw_fbo: moderngl.Framebuffer | None = None
        self.blurred_ao: moderngl.Texture | None = None
        self.blurred_fbo: moderngl.Framebuffer | None = None
        self.size = (0, 0)
        self.resize(size)

    def resize(self, size: tuple[int, int]) -> None:
        if size == self.size:
            return
        self._release()
        self.raw_ao = self.ctx.texture(size, 1, dtype="f2")
        self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_ao])
        self.blurred_ao = self.ctx.texture(size, 1, dtype="f2")
        self.blurred_fbo = self.ctx.framebuffer(color_attachments=[self.blurred_ao])
        self.size = size

    def _release(self) -> None:
        if self.raw_fbo is not None:
            self.raw_fbo.release()
            self.raw_ao.release()
            self.blurred_fbo.release()
            self.blurred_ao.release()

    def set_sample_count(self, count: int) -> None:
        self.program["u_sample_count"].value = max(0, min(count, KERNEL_SIZE))

    def render(self, prepass, proj, inv_proj) -> moderngl.Texture:
        ctx = self.ctx

        self.raw_fbo.use()
        ctx.viewport = (0, 0, *self.size)
        ctx.disable(ctx.DEPTH_TEST)
        prepass.normal.use(location=PREPASS_NORMAL_UNIT)
        prepass.depth.use(location=PREPASS_DEPTH_UNIT)
        self.noise_texture.use(location=NOISE_TEXTURE_UNIT)
        self.program["u_proj"].write(mat4_to_array(proj).tobytes())
        self.program["u_inv_proj"].write(mat4_to_array(inv_proj).tobytes())
        self.program["u_noise_scale"].value = (self.size[0] / NOISE_DIM, self.size[1] / NOISE_DIM)
        self.fullscreen.draw()

        self.blurred_fbo.use()
        ctx.viewport = (0, 0, *self.size)
        self.raw_ao.use(location=RAW_AO_UNIT)
        self.blur_fullscreen.draw()

        return self.blurred_ao
