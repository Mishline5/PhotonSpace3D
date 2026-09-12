"""Directional-light shadow map: one depth-only pass, resolution driven by quality level.

Deliberately NOT resize-dependent on the window (constraint 6 is about the
screen, not this): shadow quality is a function of scene-space texel
density, so this only changes resolution when settings.QualitySettings's
shadow_level changes, never on a window resize. Only the directional light
casts shadows - point-light shadows would need a cube map (or
dual-paraboloid) per light, real extra complexity for a second light this
minimal demo scene doesn't need; see design_report.md.

The light's view/projection is a fixed orthographic box centered on the
world origin, generously sized to cover this demo's ground plane - not
fitted to the scene's actual current bounding box each frame. A fixed box is
simpler and fully correct for a scene that stays deliberately small and
centered near the origin (documented limitation, see design_report.md).
"""
from __future__ import annotations

import glm
import moderngl

from .shader import load_program
from .gl_math import mat4_to_array

ORTHO_HALF_EXTENT = 8.0
LIGHT_DISTANCE = 15.0
NEAR, FAR = 0.5, 40.0


class ShadowPass:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.program = load_program(ctx, "shadow.vert", "shadow.frag")
        self.depth: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.map_size = 0
        self.set_resolution(1024)  # placeholder; Renderer applies the real QualitySettings value immediately

    def set_resolution(self, size: int) -> None:
        """No-op if `size` matches the current resolution or is 0 (settings
        level 0 = off; the pass simply won't render, see Renderer.render)."""
        if size <= 0 or size == self.map_size:
            return
        self._release()
        self.depth = self.ctx.depth_texture((size, size))
        self.depth.repeat_x = False
        self.depth.repeat_y = False
        self.depth.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer(color_attachments=[], depth_attachment=self.depth)
        self.map_size = size

    def _release(self) -> None:
        if self.fbo is not None:
            self.fbo.release()
            self.depth.release()

    @staticmethod
    def light_view_proj(light_direction) -> glm.mat4:
        light_dir = glm.normalize(glm.vec3(*light_direction))
        up = glm.vec3(0.0, 1.0, 0.0)
        if abs(glm.dot(light_dir, up)) > 0.99:
            up = glm.vec3(1.0, 0.0, 0.0)
        eye = -light_dir * LIGHT_DISTANCE
        view = glm.lookAt(eye, glm.vec3(0.0), up)
        proj = glm.ortho(-ORTHO_HALF_EXTENT, ORTHO_HALF_EXTENT,
                          -ORTHO_HALF_EXTENT, ORTHO_HALF_EXTENT, NEAR, FAR)
        return proj * view

    def render(self, scene, light_view_proj: glm.mat4) -> None:
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.map_size, self.map_size)
        self.fbo.clear(depth=1.0)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        lvp_bytes = mat4_to_array(light_view_proj).tobytes()
        for obj in scene.iter_shadow_casters():
            self.program["u_light_view_proj"].write(lvp_bytes)
            self.program["u_model"].write(mat4_to_array(obj.transform.matrix()).tobytes())
            obj.mesh.vertex_array(self.program).render()
