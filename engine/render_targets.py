"""Screen-sized render targets that must track window resize.

Fixed-resolution targets that must NOT track window size (the shadow map)
get their own dedicated module instead, since resizing the window must
never touch them (see design_report.md, constraint 6).
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


class MSAATarget:
    """Multisample color+depth renderbuffers the forward pass draws into
    when MSAA is enabled, resolved into an HDRTarget via `resolve_into`
    before any later pass reads it - so the prepass/SSAO/volumetric/
    tonemap/auto-exposure passes never need to know MSAA exists at all,
    they only ever see the already-resolved single-sample HDRTarget.

    Renderbuffers, not textures: nothing ever samples this directly (GL
    forbids sampling a multisample image as an ordinary sampler2D without
    dedicated multisample sampler types this engine has no other use for),
    only `copy_framebuffer` (a resolve blit) ever touches it.
    """

    def __init__(self, ctx: moderngl.Context, size: tuple[int, int], samples: int) -> None:
        self.ctx = ctx
        self.color: moderngl.Renderbuffer | None = None
        self.depth: moderngl.Renderbuffer | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.size = (0, 0)
        self.samples = 0
        self._rebuild(size, self._clamp_samples(samples))

    def _clamp_samples(self, samples: int) -> int:
        # Clamp against this GPU's actual limit (e.g. 4 on this project's own
        # M1 Max dev machine, per design_report.md) rather than trusting the
        # requested quality-level value outright - settings.py's MSAA_PARAMS
        # goes up to 8, which would otherwise ask for more samples than some
        # hardware can actually provide.
        max_samples = int(self.ctx.info.get("GL_MAX_SAMPLES") or 4)
        return max(0, min(samples, max_samples))

    def set_samples(self, samples: int) -> None:
        """No-op if the (clamped) sample count matches the current one -
        mirrors ShadowPass.set_resolution's only-reallocate-on-genuine-change
        discipline. Clamping before this comparison (not after) matters:
        comparing an unclamped request against an already-clamped
        self.samples would never be equal when the request exceeds this
        GPU's limit, reallocating the renderbuffers every single frame for
        no reason."""
        samples = self._clamp_samples(samples)
        if samples == self.samples:
            return
        self._rebuild(self.size, samples)

    def resize(self, size: tuple[int, int]) -> None:
        if size == self.size:
            return
        self._rebuild(size, self.samples)

    def _rebuild(self, size: tuple[int, int], samples: int) -> None:
        """`samples` must already be clamped - callers (set_samples/__init__)
        do that, resize() just carries the current value forward."""
        self._release()
        self.samples = samples
        self.size = size
        if samples <= 0 or size[0] == 0 or size[1] == 0:
            self.color = None
            self.depth = None
            self.fbo = None
            return
        self.color = self.ctx.renderbuffer(size, 3, samples=samples, dtype="f2")
        self.depth = self.ctx.depth_renderbuffer(size, samples=samples)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color], depth_attachment=self.depth)

    def _release(self) -> None:
        if self.fbo is not None:
            self.fbo.release()
            self.color.release()
            self.depth.release()

    def use(self) -> None:
        self.fbo.use()

    def resolve_into(self, hdr: "HDRTarget") -> None:
        # GL forbids averaging depth in a blit resolve (nearest-sample only,
        # not a bug) - only color gets true multisample averaging.
        self.ctx.copy_framebuffer(hdr.fbo, self.fbo)
