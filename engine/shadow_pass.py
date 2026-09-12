"""Directional-light shadow map: one depth-only pass, resolution driven by quality level.

Deliberately NOT resize-dependent on the window (constraint 6 is about the
screen, not this): shadow quality is a function of scene-space texel
density, so this only changes resolution when settings.QualitySettings's
shadow_level changes, never on a window resize. Only the directional light
casts shadows - point-light shadows would need a cube map (or
dual-paraboloid) per light, real extra complexity for a second light this
minimal demo scene doesn't need; see design_report.md.

The light's view/projection is now fit to the scene's own bounding SPHERE
(Scene.world_bounds(), recomputed every frame - cheap CPU matrix/vector math
at this object count) rather than a fixed constant box. A bounding sphere
(not a tight per-frame OBB) is a deliberate, documented simplification:
rotation-invariant (no popping as objects move/rotate) and simple, at the
cost of some shadow-map texel density a tighter fit would have recovered -
see design_report.md. This is a genuinely different invariant from the map
*resolution* above: the frustum's extent is recomputed every frame (cheap),
the map's pixel dimensions are not (expensive GPU realloc) - don't conflate
the two when reading this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import glm
import moderngl

from .shader import load_program
from .gl_math import mat4_to_array

# Floor under the fitted radius so a tiny/empty scene never collapses to a
# degenerate near-zero frustum (e.g. before any object has been added).
MIN_ORTHO_RADIUS = 1.0


@dataclass
class ShadowFrustum:
    view_proj: glm.mat4
    near: float
    far: float
    radius: float  # world-space bounding-sphere radius the frustum was fit to - PCSS needs this for penumbra scale


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
    def fit_frustum(light_direction, bounds_min: glm.vec3, bounds_max: glm.vec3) -> ShadowFrustum:
        """Orthographic frustum tightly enclosing the given world-space AABB's
        bounding sphere, viewed from the directional light. See the module
        docstring for why a sphere (not a tight OBB) is the deliberate choice
        here."""
        light_dir = glm.normalize(glm.vec3(*light_direction))
        up = glm.vec3(0.0, 1.0, 0.0)
        if abs(glm.dot(light_dir, up)) > 0.99:
            up = glm.vec3(1.0, 0.0, 0.0)

        center = (bounds_min + bounds_max) * 0.5
        radius = max(glm.length(bounds_max - bounds_min) * 0.5, MIN_ORTHO_RADIUS)
        light_distance = radius * 2.0 + 1.0

        eye = center - light_dir * light_distance
        view = glm.lookAt(eye, center, up)
        near = 0.1
        far = light_distance + radius + 2.0  # generous margin past the sphere's far side
        proj = glm.ortho(-radius, radius, -radius, radius, near, far)
        return ShadowFrustum(proj * view, near, far, radius)

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
