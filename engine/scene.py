"""Scene graph: plain data, so adding/moving/removing objects never touches the renderer.

`Renderer` only ever iterates `Scene.objects` (via the iter_* helpers below)
- it never special-cases "the plane" or "the cube" by name or index. Adding
a third object is a pure data change wherever the Scene is built (main.py),
with zero renderer edits, satisfying the "add/move/remove without touching
the engine core" requirement directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .transform import Transform
from .mesh import Mesh


@dataclass
class Material:
    albedo: tuple[float, float, float] = (0.8, 0.8, 0.8)
    metallic: float = 0.0
    roughness: float = 0.5
    # Scales how strongly this surface picks up hybrid ray-traced
    # reflections, independent of the roughness-driven Fresnel weighting
    # applied in the shader - lets a material opt out of RT entirely (0.0)
    # without affecting its direct PBR response.
    reflectivity: float = 1.0


@dataclass
class RTPrimitive:
    """Analytic ray-traceable shape paired 1:1 with a SceneObject.

    kind 0 = plane: bounded rectangle through the object's local origin with
             normal +Y, positioned/oriented by the object's transform.
    kind 1 = box: axis-aligned in local space with the given half-extents,
             oriented/positioned by the object's transform (so a rotated or
             scaled SceneObject still gets a correct OBB test).
    kind 2 = sphere: centered at the object's local origin, radius =
             half_extents.x, oriented/positioned by the object's transform.

    Cylinders, cones and other primitive shapes don't participate in the RT
    pass yet (no rt_primitive => simply excluded from iter_rt_primitives) -
    an intentional, documented limitation rather than an oversight; see
    design_report.md.
    """
    kind: int
    half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5)


class SceneObject:
    def __init__(self, name: str, mesh: Mesh, material: Material,
                 transform: Transform | None = None,
                 rt_primitive: RTPrimitive | None = None,
                 casts_shadow: bool = True) -> None:
        self.name = name
        self.mesh = mesh
        self.material = material
        self.transform = transform if transform is not None else Transform()
        self.rt_primitive = rt_primitive
        self.casts_shadow = casts_shadow
        self.visible = True
        # Renderer-owned: last Transform.version successfully uploaded for
        # this object. -1 guarantees the first frame always uploads once.
        self.uploaded_version = -1


@dataclass
class DirectionalLight:
    direction: tuple[float, float, float] = (-0.4, -1.0, -0.3)
    color: tuple[float, float, float] = (1.0, 0.96, 0.9)
    intensity: float = 2.0


@dataclass
class PointLight:
    position: tuple[float, float, float] = (2.2, 2.0, 1.5)
    color: tuple[float, float, float] = (0.7, 0.78, 0.95)  # subtle cool-white fill, not a saturated hue
    intensity: float = 8.0
    range: float = 10.0


class Scene:
    def __init__(self) -> None:
        self.objects: list[SceneObject] = []
        self.directional_light = DirectionalLight()
        self.point_lights: list[PointLight] = [PointLight()]

    def add(self, obj: SceneObject) -> SceneObject:
        self.objects.append(obj)
        return obj

    def remove(self, obj_or_name) -> None:
        if isinstance(obj_or_name, str):
            self.objects = [o for o in self.objects if o.name != obj_or_name]
        else:
            self.objects.remove(obj_or_name)

    def find(self, name: str) -> SceneObject | None:
        return next((o for o in self.objects if o.name == name), None)

    def iter_visible(self):
        return (o for o in self.objects if o.visible)

    def iter_shadow_casters(self):
        return (o for o in self.objects if o.visible and o.casts_shadow)

    def iter_rt_primitives(self):
        return (o for o in self.objects if o.visible and o.rt_primitive is not None)
