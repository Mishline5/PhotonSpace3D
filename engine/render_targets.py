"""Screen-sized render targets that must track window resize.

Only the HDR scene-color target lives here for now. Fixed-resolution
targets that must NOT track window size (the shadow map) get their own
dedicated module instead, since resizing the window must never touch them
(see design_report.md, constraint 6).
"""
from __future__ import annotations

import moderngl


class HDRTarget:
    """Linear HDR color + depth, rendered by the forward pass and later
    resolved to the screen by the tonemap pass.

    RGB16F (not RGBA32F/RGBA16F): no alpha channel is needed for opaque
    scene color, and 16-bit floats comfortably cover this scene's dynamic
    range (a couple of lights, no bloom accumulation pipeline) - chosen over
    the even more compact packed R11F_G11F_B10F purely because ModernGL's
    plain dtype path for it is well-documented and low-risk, whereas the
    packed format needs a raw GL internal_format override; the memory delta
    is negligible for one screen-sized target at this scene's scale.
    """

    def __init__(self, ctx: moderngl.Context, size: tuple[int, int]) -> None:
        self.ctx = ctx
        self.color: moderngl.Texture | None = None
        self.depth: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.size = (0, 0)
        self.resize(size)

    def resize(self, size: tuple[int, int]) -> None:
        if size == self.size:
            return
        self._release()
        self.color = self.ctx.texture(size, 3, dtype="f2")
        self.color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth = self.ctx.depth_texture(size)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color], depth_attachment=self.depth)
        self.size = size

    def _release(self) -> None:
        if self.fbo is not None:
            self.fbo.release()
            self.color.release()
            self.depth.release()

    def use(self) -> None:
        self.fbo.use()
