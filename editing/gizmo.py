"""Transform gizmo: builds the translate/rotate/scale handle meshes once,
computes their screen-constant apparent size, and draws them over the
already-rendered scene. Never touches engine/renderer.py - this is a second,
separate draw call issued by edit.py after Renderer.render() returns.
"""
from __future__ import annotations

import math
from pathlib import Path

import glm
import moderngl

from engine.camera import Camera
from engine.mesh import Mesh
from engine.scene import SceneObject
from engine.shader import load_program
from engine.gl_math import mat4_to_array

from .gizmo_geometry import (
    make_arrow, make_ring, make_scale_handle, axis_align_rotation, AXIS_COLOR, HIGHLIGHT_COLOR,
)

SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def gizmo_world_size(camera: Camera, object_position: glm.vec3,
                      viewport_height: int, target_px: float = 90.0) -> float:
    """World-space size that projects to `target_px` screen pixels at
    `object_position`'s distance from the camera - the exact perspective
    relationship (not a distance*constant approximation), so the gizmo
    stays a constant apparent size regardless of FOV or window size."""
    distance = glm.length(glm.vec3(camera.position) - object_position)
    return target_px * 2.0 * distance * math.tan(glm.radians(camera.fov_degrees) * 0.5) / max(viewport_height, 1)


class Gizmo:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.program = load_program(ctx, "gizmo.vert", "gizmo.frag", shader_dir=SHADER_DIR)
        self.meshes = {
            "translate": Mesh(ctx, *make_arrow()),
            "rotate": Mesh(ctx, *make_ring()),
            "scale": Mesh(ctx, *make_scale_handle()),
        }

    def handle_matrix(self, obj: SceneObject, axis: int, mode: str, apparent_size: float) -> glm.mat4:
        """Move/rotate handles are world-axis-aligned; scale handles follow
        the object's own rotation (Transform.scale applies in T*R*S order,
        before any world-space reorientation, so it's intrinsically a local-
        space quantity - see transform.py) - a deliberate asymmetry, not an
        inconsistency."""
        position = glm.vec3(obj.transform.position)
        if mode == "scale":
            rotation = obj.transform.rotation * axis_align_rotation(axis)
        else:
            rotation = axis_align_rotation(axis)
        t = glm.translate(glm.mat4(1.0), position)
        r = glm.mat4_cast(rotation)
        s = glm.scale(glm.mat4(1.0), glm.vec3(apparent_size))
        return t * r * s

    def draw(self, camera: Camera, obj: SceneObject, mode: str,
              hovered_axis: int | None, active_axis: int | None, viewport_size: tuple[int, int]) -> None:
        # No depth testing here, deliberately: ctx.screen's depth buffer is
        # never actually populated with the 3D scene's depth - the whole
        # pipeline renders offscreen (HDRTarget/MSAATarget) and only the
        # tonemap pass's flat fullscreen triangle ever touches ctx.screen,
        # with depth testing disabled and nothing written (see
        # renderer.py's tonemap step) - so testing the gizmo against it
        # would compare against stale/undefined data, not real scene depth
        # (confirmed by reading it back: the gizmo simply failed to draw at
        # all with depth testing on). Three axis handles fanning out from a
        # shared origin rarely overlap on screen anyway, so drawing them
        # unconditionally on top, in a fixed order, is a fine simplification
        # rather than reintroducing a real depth buffer just for this.
        ctx = self.ctx
        apparent_size = gizmo_world_size(camera, glm.vec3(obj.transform.position), viewport_size[1])
        view_proj = camera.projection_matrix() * camera.view_matrix()
        vao = self.meshes[mode].vertex_array(self.program)
        for axis in (0, 1, 2):
            model = self.handle_matrix(obj, axis, mode, apparent_size)
            mvp = view_proj * model
            self.program["u_mvp"].write(mat4_to_array(mvp).tobytes())
            if active_axis == axis:
                self.program["u_color"].value = HIGHLIGHT_COLOR
                self.program["u_highlight"].value = 1.0
            else:
                self.program["u_color"].value = AXIS_COLOR[axis]
                self.program["u_highlight"].value = 1.4 if hovered_axis == axis else 1.0
            vao.render()
