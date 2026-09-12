"""Mouse-ray object picking.

Deliberately duplicates the analytic intersection tests already written in
GLSL (engine/shaders/forward.frag's intersect_plane_local/intersect_box_local/
intersect_sphere_local/intersect_cylinder_local) as plain Python/PyGLM
functions - there is no mechanism to share source between a GLSL fragment
shader and Python, so this is a controlled duplication, not an oversight:
both copies follow the same structure, instruction for instruction, so a
change to one is easy to cross-check against the other. RTPrimitive
(kind/half_extents, engine/scene.py) is the actual shared source of truth
between them; only the intersection *code* is duplicated, never the data.
"""
from __future__ import annotations

import math

import glm

from engine.camera import Camera
from engine.scene import Scene, SceneObject, RTPrimitive


def build_pick_ray(camera: Camera, mouse_pos: tuple[float, float],
                    viewport_size: tuple[int, int]) -> tuple[glm.vec3, glm.vec3]:
    """Camera-space ray through the given screen-space pixel. `mouse_pos` is
    in the same (GLFW cursor-callback) coordinate space `Window.input.mouse_pos`
    already uses, paired with `viewport_size` the same way engine/settings_ui.py
    already pairs mouse_pos with framebuffer_size for ImGui - reusing that
    exact (uncorrected-for-HiDPI-scale) convention rather than introducing a
    second, inconsistent one. Known, accepted limitation, not fixed here."""
    w, h = viewport_size
    ndc_x = (mouse_pos[0] / max(w, 1)) * 2.0 - 1.0
    ndc_y = 1.0 - (mouse_pos[1] / max(h, 1)) * 2.0
    inv_view_proj = glm.inverse(camera.projection_matrix() * camera.view_matrix())
    clip = glm.vec4(ndc_x, ndc_y, 0.0, 1.0)
    world = inv_view_proj * clip
    world_point = glm.vec3(world) / world.w
    origin = glm.vec3(camera.position)
    direction = glm.normalize(world_point - origin)
    return origin, direction


def intersect_plane_local(ro: glm.vec3, rd: glm.vec3,
                           half_extents_xz: tuple[float, float]) -> float | None:
    if abs(rd.y) < 1e-6:
        return None
    t = -ro.y / rd.y
    if t <= 1e-4:
        return None
    hit = ro + rd * t
    if abs(hit.x) > half_extents_xz[0] or abs(hit.z) > half_extents_xz[1]:
        return None
    return t


def intersect_box_local(ro: glm.vec3, rd: glm.vec3, half_extents: glm.vec3) -> float | None:
    inv_rd = glm.vec3(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z)
    t0 = (-half_extents - ro) * inv_rd
    t1 = (half_extents - ro) * inv_rd
    t_near = max(min(t0.x, t1.x), min(t0.y, t1.y), min(t0.z, t1.z))
    t_far = min(max(t0.x, t1.x), max(t0.y, t1.y), max(t0.z, t1.z))
    if t_near > t_far or t_far <= 1e-4:
        return None
    return t_near if t_near > 1e-4 else t_far


def intersect_sphere_local(ro: glm.vec3, rd: glm.vec3, radius: float) -> float | None:
    b = glm.dot(ro, rd)
    c = glm.dot(ro, ro) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    t = -b - sqrt_disc
    if t <= 1e-4:
        t = -b + sqrt_disc
    if t <= 1e-4:
        return None
    return t


def intersect_cylinder_local(ro: glm.vec3, rd: glm.vec3, radius: float, half_height: float) -> float | None:
    """Mirrors forward.frag's intersect_cylinder_local - see that function's
    comment for why both roots of the side quadratic, plus both caps, are
    each checked independently rather than picking the nearer root first."""
    closest_t = None

    a = rd.x * rd.x + rd.z * rd.z
    if a > 1e-8:
        b = 2.0 * (ro.x * rd.x + ro.z * rd.z)
        c = ro.x * ro.x + ro.z * ro.z - radius * radius
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            for t in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
                if t > 1e-4 and (closest_t is None or t < closest_t):
                    hit = ro + rd * t
                    if abs(hit.y) <= half_height:
                        closest_t = t

    if abs(rd.y) > 1e-6:
        for cap_y in (-half_height, half_height):
            t = (cap_y - ro.y) / rd.y
            if t > 1e-4 and (closest_t is None or t < closest_t):
                hit = ro + rd * t
                if hit.x * hit.x + hit.z * hit.z <= radius * radius:
                    closest_t = t

    return closest_t


def _intersect(rt: RTPrimitive, ro: glm.vec3, rd: glm.vec3) -> float | None:
    if rt.kind == 0:
        return intersect_plane_local(ro, rd, (rt.half_extents[0], rt.half_extents[2]))
    if rt.kind == 2:
        return intersect_sphere_local(ro, rd, rt.half_extents[0])
    if rt.kind == 3:
        return intersect_cylinder_local(ro, rd, rt.half_extents[0], rt.half_extents[1])
    return intersect_box_local(ro, rd, glm.vec3(*rt.half_extents))


def pick_object(scene: Scene, ray_origin: glm.vec3, ray_dir: glm.vec3) -> SceneObject | None:
    """Nearest object (by RTPrimitive, same eligibility as the hybrid RT
    pass's reflection list - see ensure_pickable for objects that don't
    have one) hit by the given world-space ray."""
    closest_t = None
    closest_obj = None
    for obj in scene.iter_rt_primitives():
        inv_model = glm.inverse(obj.transform.matrix())
        ro = glm.vec3(inv_model * glm.vec4(ray_origin, 1.0))
        rd = glm.normalize(glm.vec3(inv_model * glm.vec4(ray_dir, 0.0)))
        t = _intersect(obj.rt_primitive, ro, rd)
        if t is not None and (closest_t is None or t < closest_t):
            closest_t = t
            closest_obj = obj
    return closest_obj


def ensure_pickable(scene: Scene) -> None:
    """Safety net: assigns a box RTPrimitive (a reasonable bounding-box
    approximation, not necessarily exact) to any visible object that
    doesn't already have one, so nothing added to a scene without RT/
    reflection support in mind becomes silently unselectable. Scene-
    construction-time decision, lives in editing/ rather than engine/ - it
    changes what a scene exposes to the *tool*, not what the renderer does
    with it (an object gained this way also becomes RT-reflective, an
    accepted side effect, not a new limitation: see design_report.md)."""
    for obj in scene.iter_visible():
        if obj.rt_primitive is not None:
            continue
        lo, hi = obj.mesh.local_min, obj.mesh.local_max
        half_extents = (
            float(max(hi[0] - lo[0], 1e-3) * 0.5),
            float(max(hi[1] - lo[1], 1e-3) * 0.5),
            float(max(hi[2] - lo[2], 1e-3) * 0.5),
        )
        obj.rt_primitive = RTPrimitive(kind=1, half_extents=half_extents)
