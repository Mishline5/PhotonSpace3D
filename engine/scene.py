"""Scene graph: plain data, so adding/moving/removing objects never touches the renderer.

`Renderer` only ever iterates `Scene.objects` (via the iter_* helpers below)
- it never special-cases "the plane" or "the cube" by name or index. Adding
a third object is a pure data change wherever the Scene is built (main.py),
with zero renderer edits, satisfying the "add/move/remove without touching
the engine core" requirement directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import glm
import moderngl

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
    # Optional per-pixel maps (engine/procedural_textures.py generates these
    # procedurally - never a file loaded from disk). None means "use the
    # scalar fields above" - Renderer falls back to a 1x1 texture encoding
    # that scalar exactly (engine/material_textures.py) so forward.frag can
    # stay on a single "always sample a texture" code path. Left as plain
    # moderngl.Texture handles (not e.g. a file path) since Material is
    # otherwise plain, ctx-free data - whatever builds the Scene is
    # responsible for creating these on the right GL context.
    albedo_map: moderngl.Texture | None = None
    roughness_map: moderngl.Texture | None = None
    metallic_map: moderngl.Texture | None = None
    normal_map: moderngl.Texture | None = None


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
    kind 3 = cylinder: axis along the object's local +Y, radius =
             half_extents.x, half-height = half_extents.y, capped flat at
             both ends (matches primitives.make_cylinder's geometry).

    Cones and other primitive shapes don't participate in the RT pass yet
    (no rt_primitive => simply excluded from iter_rt_primitives) - an
    intentional, documented limitation rather than an oversight; see
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
class AmbientLight:
    """Two-color hemisphere standing in for indirect/IBL lighting - replaces
    the old flat `vec3(0.05)*albedo` constant with something that at least
    varies by surface orientation (brighter facing up toward the sky,
    darker facing down toward the ground bounce), still far short of a real
    environment capture but a real, if cheap, physical improvement over a
    constant. See forward.frag's `ambient_hemisphere`."""
    # Tuned to roughly match the flat vec3(0.05) constant this replaced in
    # total brightness (the directional light's intensity=2.0 was already
    # balanced against that old, much dimmer constant) while still adding
    # real sky/ground directionality - a brighter hemisphere reads as more
    # "correct" in isolation but pushes every surface further up the ACES
    # curve's compressive shoulder, crushing exactly the kind of subtle
    # albedo/roughness texture variation this pass exists to add (found by
    # comparing renders at default vs. a much lower exposure - the texture
    # detail was always there, just tonemapped into invisibility).
    sky_color: tuple[float, float, float] = (0.12, 0.13, 0.15)
    ground_color: tuple[float, float, float] = (0.05, 0.045, 0.04)
    intensity: float = 1.0


@dataclass
class DirectionalLight:
    direction: tuple[float, float, float] = (-0.4, -1.0, -0.3)
    color: tuple[float, float, float] = (1.0, 0.96, 0.9)
    intensity: float = 2.0
    # Dimensionless "how big is this light" dial driving PCSS penumbra width
    # (see shadow_pass.py/forward.frag) - not a physical angular size, just
    # an artistic knob in the same spirit as intensity/color above it.
    softness: float = 0.15


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
        self.ambient_light = AmbientLight()

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

    def world_bounds(self) -> tuple[glm.vec3, glm.vec3]:
        """Union AABB (world space) of every visible object's 8 transformed
        local-space corners - used to fit the directional shadow frustum to
        what's actually in the scene (shadow_pass.fit_frustum) instead of a
        fixed constant. Falls back to a small box around the origin for an
        empty scene so callers never special-case a degenerate bound."""
        corners_world = []
        for obj in self.iter_visible():
            model = obj.transform.matrix()
            lo, hi = obj.mesh.local_min, obj.mesh.local_max
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        corners_world.append(glm.vec3(model * glm.vec4(x, y, z, 1.0)))

        if not corners_world:
            return glm.vec3(-1.0, -1.0, -1.0), glm.vec3(1.0, 1.0, 1.0)

        bounds_min = glm.vec3(corners_world[0])
        bounds_max = glm.vec3(corners_world[0])
        for c in corners_world[1:]:
            bounds_min = glm.min(bounds_min, c)
            bounds_max = glm.max(bounds_max, c)
        return bounds_min, bounds_max
