"""Procedural gizmo geometry: translate arrows, rotate rings, scale handles.

Built from engine/primitives.py's own generators (cylinder/cone/cube) rather
than a bespoke low-level mesh format - a gizmo arrow really is just a
cylinder shaft with a cone head, so reusing those generators (translated
into place, then concatenated) avoids a second geometry-building path for
what's structurally the same kind of shape.
"""
from __future__ import annotations

import numpy as np
import glm

from engine.mesh import VERTEX_DTYPE
from engine.primitives import make_cylinder, make_cone, make_cube

AXIS_WORLD_DIR = {0: glm.vec3(1.0, 0.0, 0.0), 1: glm.vec3(0.0, 1.0, 0.0), 2: glm.vec3(0.0, 0.0, 1.0)}
# Standard red/green/blue = X/Y/Z convention (Blender/Unity/Unreal) - the one
# place in this project where a saturated color is correct, not a "flashy
# surface" violation: this is tool chrome, not scene material (see
# design_report.md's material-realism passage for the constraint this is
# deliberately exempt from).
AXIS_COLOR = {0: (0.85, 0.15, 0.15), 1: (0.15, 0.75, 0.15), 2: (0.15, 0.45, 0.9)}
HIGHLIGHT_COLOR = (0.95, 0.82, 0.15)


def axis_align_rotation(axis: int) -> glm.quat:
    """Rotation mapping the generators' own local +Y (their natural "up") onto
    world axis `axis`, so make_arrow/make_ring/make_scale_handle only ever
    need to be built once, along Y, then reoriented per axis at draw/pick
    time rather than regenerated three times."""
    if axis == 0:
        return glm.angleAxis(glm.radians(-90.0), glm.vec3(0.0, 0.0, 1.0))
    if axis == 2:
        return glm.angleAxis(glm.radians(90.0), glm.vec3(1.0, 0.0, 0.0))
    return glm.quat(1.0, 0.0, 0.0, 0.0)


def _translate_y(vertices: np.ndarray, dy: float) -> np.ndarray:
    out = vertices.copy()
    out["position"][:, 1] += dy
    return out


def _concat(mesh_a: tuple[np.ndarray, np.ndarray], mesh_b: tuple[np.ndarray, np.ndarray]
            ) -> tuple[np.ndarray, np.ndarray]:
    va, ia = mesh_a
    vb, ib = mesh_b
    vertices = np.concatenate([va, vb])
    indices = np.concatenate([ia, ib.astype(ia.dtype) + len(va)])
    return vertices, indices


#  Both handle generators below sum shaft+head to exactly 1.0 local units
# (matching make_ring's radius=1.0 default) so a single `apparent_size`
# scale factor (see gizmo.py's gizmo_world_size) means the same thing -
# "this handle's overall reach in world units" - across every mode, which
# gizmo_interaction.py's hover math relies on.
def make_arrow(shaft_radius: float = 0.018, shaft_length: float = 0.75,
               head_radius: float = 0.05, head_length: float = 0.25,
               segments: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Translate-gizmo handle: a cylinder shaft from the origin up to
    `shaft_length`, capped with a cone head - the standard arrow shape,
    built entirely along local +Y (see axis_align_rotation)."""
    shaft = make_cylinder(shaft_radius, shaft_length, segments)
    shaft = (_translate_y(shaft[0], shaft_length * 0.5), shaft[1])
    head = make_cone(head_radius, head_length, segments)
    head = (_translate_y(head[0], shaft_length + head_length * 0.5), head[1])
    return _concat(shaft, head)


def make_scale_handle(shaft_radius: float = 0.018, shaft_length: float = 0.75,
                       head_half_extent: float = 0.05, segments: int = 14
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Scale-gizmo handle: same shaft as make_arrow, a small cube instead of
    a cone at the tip (Blender's own convention for telling scale handles
    apart from translate handles at a glance)."""
    shaft = make_cylinder(shaft_radius, shaft_length, segments)
    shaft = (_translate_y(shaft[0], shaft_length * 0.5), shaft[1])
    head = make_cube(head_half_extent)
    head = (_translate_y(head[0], shaft_length + head_half_extent), head[1])
    return _concat(shaft, head)


def make_ring(radius: float = 1.0, tube_radius: float = 0.02,
              radial_segments: int = 32, tube_segments: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Rotate-gizmo handle: a real triangulated torus around local Y (its
    "hole" faces local +Y), not a GL_LINE_STRIP - glLineWidth > 1 isn't
    reliably supported in core profile (often silently clamped to 1.0,
    including on macOS - the same class of portability trap design_report.md
    documents elsewhere for this project), so a thin torus is what actually
    guarantees a visible ring on every platform this engine targets."""
    verts = []
    for i in range(radial_segments + 1):
        theta = i * 2.0 * np.pi / radial_segments
        r_hat = np.array([np.cos(theta), 0.0, np.sin(theta)])
        center = r_hat * radius
        for j in range(tube_segments + 1):
            phi = j * 2.0 * np.pi / tube_segments
            offset = np.cos(phi) * r_hat + np.sin(phi) * np.array([0.0, 1.0, 0.0])
            pos = center + offset * tube_radius
            verts.append((tuple(pos), tuple(offset), (i / radial_segments, j / tube_segments)))

    vertices = np.array(verts, dtype=VERTEX_DTYPE)
    cols = tube_segments + 1
    indices = []
    for i in range(radial_segments):
        for j in range(tube_segments):
            a = i * cols + j
            b = a + 1
            c = a + cols
            d = c + 1
            indices.extend((a, c, d))
            indices.extend((a, d, b))
    return vertices, np.array(indices, dtype=np.uint32)
