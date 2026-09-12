"""Attributeless fullscreen-triangle draw, shared by post-process passes
(tonemap now; SSAO later) - see shaders/fullscreen.vert for the no-VBO trick."""
from __future__ import annotations

import moderngl


class FullscreenPass:
    def __init__(self, ctx: moderngl.Context, program: moderngl.Program) -> None:
        self.ctx = ctx
        self.program = program
        self.vao = ctx.vertex_array(program, [])

    def draw(self) -> None:
        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)
