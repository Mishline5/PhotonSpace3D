"""Factory for addable scene objects: the single place both the demo scene
(main.py) and the editor's Add menu build a cube / sphere / cylinder / cone /
plane / point-light / spot-light, so the two never drift apart.

Meshes are cached per (ctx, kind) so adding ten cubes uploads one cube mesh,
not ten (geometry is immutable, see mesh.py). Lights are ordinary
SceneObjects carrying a LightComponent plus a small emissive marker mesh so
you can see and click them; the marker is editor-only chrome (SceneObject
.marker), hidden when the scene runs in game mode.
"""
from __future__ import annotations

import glm

from .mesh import Mesh
from .primitives import make_cube, make_uv_sphere, make_cylinder, make_cone, make_plane
from .scene import (
    SceneObject, Material, RTPrimitive, LightComponent, Transform,
    LIGHT_POINT, LIGHT_SPOT,
)

# Addable geometry kinds shown in the editor Add menu, with the RTPrimitive
# used for picking/reflection and a neutral default material tone.
GEOMETRY_KINDS = ("cube", "sphere", "cylinder", "cone", "plane")
LIGHT_KINDS = ("point_light", "spot_light")

_NEUTRAL = (0.60, 0.60, 0.62)

_mesh_cache: dict[tuple[int, str], Mesh] = {}


def _mesh(ctx, kind: str) -> Mesh:
    key = (id(ctx), kind)
    m = _mesh_cache.get(key)
    if m is None:
        if kind == "cube":
            m = Mesh(ctx, *make_cube(0.5))
        elif kind == "sphere":
            m = Mesh(ctx, *make_uv_sphere(0.5))
        elif kind == "cylinder":
            m = Mesh(ctx, *make_cylinder(0.4, 1.0))
        elif kind == "cone":
            m = Mesh(ctx, *make_cone(0.5, 1.0))
        elif kind == "plane":
            m = Mesh(ctx, *make_plane(4.0))
        elif kind == "light_bulb":
            m = Mesh(ctx, *make_uv_sphere(0.12, segments=16, rings=8))
        elif kind == "light_cone":
            m = Mesh(ctx, *make_cone(0.16, 0.32, segments=16))
        else:
            raise ValueError(f"unknown mesh kind {kind!r}")
        _mesh_cache[key] = m
    return m


def _rt_for(kind: str) -> RTPrimitive:
    if kind == "sphere":
        return RTPrimitive(kind=2, half_extents=(0.5, 0.0, 0.0))
    if kind == "cylinder":
        return RTPrimitive(kind=3, half_extents=(0.4, 0.5, 0.0))
    if kind == "plane":
        return RTPrimitive(kind=0, half_extents=(2.0, 0.0, 2.0))
    if kind == "cone":
        # No analytic cone RT test; a box bound keeps it pickable.
        return RTPrimitive(kind=1, half_extents=(0.5, 0.5, 0.5))
    return RTPrimitive(kind=1, half_extents=(0.5, 0.5, 0.5))  # cube


def make_geometry(ctx, kind: str, scene, position=(0.0, 0.0, 0.0)) -> SceneObject:
    mesh = _mesh(ctx, kind)
    obj = SceneObject(
        scene.unique_name(kind), mesh,
        Material(albedo=_NEUTRAL, metallic=0.0, roughness=0.6),
        rt_primitive=_rt_for(kind),
    )
    obj.transform.position = position
    return obj


def make_light(ctx, kind: str, scene, position=(0.0, 2.5, 0.0)) -> SceneObject:
    """kind is "point_light" or "spot_light"."""
    is_spot = kind == "spot_light"
    color = (1.0, 1.0, 1.0)
    mesh = _mesh(ctx, "light_cone" if is_spot else "light_bulb")
    mat = Material(albedo=color, metallic=0.0, roughness=1.0, reflectivity=0.0, emissive=color)
    light = LightComponent(type=LIGHT_SPOT if is_spot else LIGHT_POINT, color=color, intensity=10.0)
    name = scene.unique_name("spot" if is_spot else "point")
    obj = SceneObject(
        name, mesh, mat,
        rt_primitive=RTPrimitive(kind=1, half_extents=(0.18, 0.18, 0.18)),
        casts_shadow=False, light=light, collision_enabled=False, marker=True,
    )
    obj.transform.position = position
    return obj


def sync_light_marker(obj: SceneObject) -> None:
    """Keep a light marker's emissive glow matching its light colour (call
    after the Properties panel edits the colour)."""
    if obj.light is not None and obj.material is not None:
        obj.material.emissive = tuple(obj.light.color)
        obj.material.albedo = tuple(obj.light.color)
