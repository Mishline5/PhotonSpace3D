"""Depth + view-space normal prepass, feeding SSAO before the forward pass runs.

Screen-sized (resize-dependent), unlike the shadow map. This intentionally
is NOT a full G-buffer (no albedo/material channels): with the whole scene
still shaded in one forward pass, this prepass exists only to give SSAO
something to reconstruct position/normal from before the forward pass has
run. It duplicates some depth computation the forward pass also does - the
standard, accepted cost of prepass-driven SSAO.
"""
from __future__ import annotations

import moderngl

from .shader import load_program
from .frame_ubo import FrameUBO
from .gl_math import mat4_to_array, mat3_to_array, normal_matrix


class PrepassTarget:
    def __init__(self, ctx: moderngl.Context, size: tuple[int, int]) -> None:
        self.ctx = ctx
        self.program = load_program(ctx, "prepass.vert", "prepass.frag")
        FrameUBO.bind_to_program(self.program)
        self.normal: moderngl.Texture | None = None
        self.depth: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.size = (0, 0)
        self.resize(size)

    def resize(self, size: tuple[int, int]) -> None:
        if size == self.size:
            return
        self._release()
        self.normal = self.ctx.texture(size, 3, dtype="f2")
        self.depth = self.ctx.depth_texture(size)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.normal], depth_attachment=self.depth)
        self.size = size

    def _release(self) -> None:
        if self.fbo is not None:
            self.fbo.release()
            self.normal.release()
            self.depth.release()

    def render(self, scene) -> None:
        self.fbo.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.fbo.clear(depth=1.0)
        prog = self.program
        for obj in scene.iter_visible():
            model = obj.transform.matrix()
            prog["u_model"].write(mat4_to_array(model).tobytes())
            prog["u_normal_matrix"].write(mat3_to_array(normal_matrix(model)).tobytes())
            obj.mesh.vertex_array(prog).render()
