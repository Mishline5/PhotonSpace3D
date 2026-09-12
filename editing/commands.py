"""Editor commands: add / delete / copy / paste objects and lights.

All scene mutations the toolbar and context menu perform live here, so the UI
stays declarative (a button calls a command) and the session just holds the
clipboard. Copy takes a snapshot at copy-time (so later edits to the source
don't leak into a pending paste); paste rebuilds a fully independent object -
sharing only the immutable Mesh (safe, geometry never changes) and
regenerating its own texture maps so releasing one object's maps never
dangles another's.
"""
from __future__ import annotations

import glm

from engine.scene import Scene, SceneObject, Material, RTPrimitive, LightComponent
from engine.transform import Transform
from engine import prefabs
from engine.material_textures import apply_texture_kind


def spawn_position(camera, distance: float = 4.0) -> tuple[float, float, float]:
    """A point in front of the camera, dropped near ground level so a new
    object appears where the user is looking rather than at the origin."""
    p = glm.vec3(camera.position) + glm.vec3(camera.front) * distance
    p.y = max(p.y, 0.5)
    return (float(p.x), float(p.y), float(p.z))


def add_geometry(ctx, scene: Scene, kind: str, camera) -> SceneObject:
    obj = prefabs.make_geometry(ctx, kind, scene, position=spawn_position(camera))
    return scene.add(obj)


def add_light(ctx, scene: Scene, kind: str, camera) -> SceneObject:
    pos = spawn_position(camera, distance=3.0)
    pos = (pos[0], max(pos[1], 2.5), pos[2])
    obj = prefabs.make_light(ctx, kind, scene, position=pos)
    return scene.add(obj)


def delete_object(scene: Scene, obj: SceneObject) -> None:
    scene.remove(obj)


def snapshot(obj: SceneObject) -> dict:
    """Copy-time snapshot of everything paste needs to rebuild an independent
    twin (shares only the immutable mesh)."""
    t = obj.transform
    mat = obj.material
    return {
        "mesh": obj.mesh,
        "name_base": obj.name.split(".")[0],
        "position": (float(t.position.x), float(t.position.y), float(t.position.z)),
        "rotation": glm.quat(t.rotation),
        "scale": (float(t.scale.x), float(t.scale.y), float(t.scale.z)),
        "material": {
            "albedo": tuple(mat.albedo), "metallic": mat.metallic,
            "roughness": mat.roughness, "reflectivity": mat.reflectivity,
            "texture_kind": mat.texture_kind, "noise_grain": mat.noise_grain,
            "noise_strength": mat.noise_strength,
            "emissive": tuple(mat.emissive) if mat.emissive is not None else None,
        },
        "rt": (obj.rt_primitive.kind, tuple(obj.rt_primitive.half_extents)) if obj.rt_primitive else None,
        "light": (obj.light.type, tuple(obj.light.color), obj.light.intensity, obj.light.spot_angle_deg)
                 if obj.light else None,
        "casts_shadow": obj.casts_shadow,
        "collision_enabled": obj.collision_enabled,
        "marker": obj.marker,
    }


def paste(ctx, scene: Scene, snap: dict, offset: float = 0.8) -> SceneObject:
    m = snap["material"]
    mat = Material(
        albedo=m["albedo"], metallic=m["metallic"], roughness=m["roughness"],
        reflectivity=m["reflectivity"], texture_kind=m["texture_kind"],
        noise_grain=m["noise_grain"], noise_strength=m["noise_strength"],
        emissive=m["emissive"],
    )
    transform = Transform(
        position=(snap["position"][0] + offset, snap["position"][1], snap["position"][2] + offset),
        scale=snap["scale"],
    )
    transform.rotation = glm.quat(snap["rotation"])
    rt = RTPrimitive(kind=snap["rt"][0], half_extents=snap["rt"][1]) if snap["rt"] else None
    light = None
    if snap["light"] is not None:
        lt, col, inten, ang = snap["light"]
        light = LightComponent(type=lt, color=col, intensity=inten, spot_angle_deg=ang)
    obj = SceneObject(
        scene.unique_name(snap["name_base"]), snap["mesh"], mat, transform=transform,
        rt_primitive=rt, casts_shadow=snap["casts_shadow"], light=light,
        collision_enabled=snap["collision_enabled"], marker=snap["marker"],
    )
    # Rebuild this twin's own texture maps (never share the source's handles).
    if mat.texture_kind == "noise":
        apply_texture_kind(ctx, mat)
    return scene.add(obj)
