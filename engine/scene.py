"""Scene graph: plain data, so adding/moving/removing objects never touches the renderer.

`Renderer` only ever iterates `Scene.objects` (via the iter_* helpers below)
- it never special-cases "the plane" or "the cube" by name or index. Adding
a third object is a pure data change wherever the Scene is built (main.py),
with zero renderer edits, satisfying the "add/move/remove without touching
the engine core" requirement directly.
"""
from __future__ import annotations

import math
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
    # stay on a single "always sample a texture" code path.
    albedo_map: moderngl.Texture | None = None
    roughness_map: moderngl.Texture | None = None
    metallic_map: moderngl.Texture | None = None
    normal_map: moderngl.Texture | None = None
    # Editor-facing texture description so the Properties panel can offer a
    # simple None / Noise choice with a couple of tweakable params, and
    # regenerate the maps above from the current albedo on demand (see
    # material_textures.apply_texture_kind). Not read by the renderer.
    texture_kind: str = "none"      # "none" | "noise"
    noise_grain: float = 8.0        # lattice cell count (higher = finer)
    noise_strength: float = 1.0
    # Unlit self-illumination added on top of shading (used for light-source
    # markers so you can see where a light is). None = not emissive.
    emissive: tuple[float, float, float] | None = None

    def has_textures(self) -> bool:
        return self.albedo_map is not None


# Light types stored in a LightComponent / consumed by lights.LightsUBO.
LIGHT_POINT = 0
LIGHT_SPOT = 1
# A spot's local emission axis before the object's rotation is applied:
# straight down, so a spot dropped above the scene lights the floor and is
# then aimed by rotating the object (its cone marker mesh points down too).
SPOT_LOCAL_DIR = glm.vec3(0.0, -1.0, 0.0)


@dataclass
class LightComponent:
    """A light carried by a SceneObject (so lights are moved/rotated/scaled
    and selected exactly like any other object). Point and spot only - the
    sun is the separate DirectionalLight. Deliberately minimal per the tool's
    "lights only have color + intensity (+ spot: aim via rotation, field via
    scale)" spec: `intensity` sets both reach and brightness."""
    type: int = LIGHT_POINT
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 10.0
    spot_angle_deg: float = 32.0   # spot outer half-angle at scale 1


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
                 casts_shadow: bool = True,
                 light: LightComponent | None = None,
                 collision_enabled: bool = True,
                 marker: bool = False) -> None:
        self.name = name
        self.mesh = mesh
        self.material = material
        self.transform = transform if transform is not None else Transform()
        self.rt_primitive = rt_primitive
        self.casts_shadow = casts_shadow
        # A light source carried by this object (None for ordinary geometry).
        self.light = light
        # Whether the game-mode character collides with this object's mesh
        # (per-object toggle, editable in the Properties panel; ignored by
        # the editor's free-fly camera which never collides).
        self.collision_enabled = collision_enabled
        # A `marker` object is editor-only chrome (e.g. a light's little glowing
        # bulb/cone so you can see + click it) - drawn in the editor, hidden
        # when the scene runs in game mode, and never collided with.
        self.marker = marker
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
    # Lowered a touch from the first texturing pass so shadowed regions
    # (lit only by this ambient term) read deeper without crushing detail -
    # paired with a slightly stronger sun for more lit/shadow contrast.
    sky_color: tuple[float, float, float] = (0.095, 0.105, 0.125)
    ground_color: tuple[float, float, float] = (0.038, 0.034, 0.030)
    intensity: float = 1.0


@dataclass
class DirectionalLight:
    direction: tuple[float, float, float] = (-0.4, -1.0, -0.3)
    color: tuple[float, float, float] = (1.0, 0.96, 0.9)
    intensity: float = 2.8
    # Dimensionless "how big is this light" dial driving PCSS penumbra width
    # (see shadow_pass.py/forward.frag) - not a physical angular size, just
    # an artistic knob. Lowered for tighter, deeper shadow edges.
    softness: float = 0.08

    def set_from_angles(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Aim the sun from a compass azimuth (rotation around the scene) and
        an elevation (height in the sky), the two dials the sun panel exposes.
        elevation 90 = straight down from zenith, ~5 = near the horizon."""
        az = math.radians(azimuth_deg)
        el = math.radians(max(elevation_deg, 1.0))
        # Direction the light *travels* (down into the scene).
        dx = math.cos(el) * math.cos(az)
        dz = math.cos(el) * math.sin(az)
        dy = -math.sin(el)
        self.direction = (dx, dy, dz)


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
        # Legacy convenience list, still honoured by collect_lights(); the
        # editor adds lights as SceneObjects with a LightComponent instead.
        self.point_lights: list[PointLight] = []
        self.ambient_light = AmbientLight()

    def add(self, obj: SceneObject) -> SceneObject:
        self.objects.append(obj)
        return obj

    def remove(self, obj_or_name) -> None:
        if isinstance(obj_or_name, str):
            self.objects = [o for o in self.objects if o.name != obj_or_name]
        else:
            self.objects.remove(obj_or_name)

    def unique_name(self, base: str) -> str:
        """A name not already used by any object (base, base.001, ...)."""
        if self.find(base) is None:
            return base
        i = 1
        while self.find(f"{base}.{i:03d}") is not None:
            i += 1
        return f"{base}.{i:03d}"

    def find(self, name: str) -> SceneObject | None:
        return next((o for o in self.objects if o.name == name), None)

    def iter_visible(self):
        return (o for o in self.objects if o.visible)

    def iter_shadow_casters(self):
        return (o for o in self.objects if o.visible and o.casts_shadow)

    def iter_rt_primitives(self):
        return (o for o in self.objects if o.visible and o.rt_primitive is not None)

    def collect_lights(self) -> list[dict]:
        """Flattened world-space light list for lights.LightsUBO. Gathers
        both legacy Scene.point_lights and every visible object carrying a
        LightComponent (a point emits omnidirectionally; a spot's aim comes
        from the object's rotation and its cone width from the object's
        scale)."""
        out: list[dict] = []
        for pl in self.point_lights:
            out.append({
                "position": tuple(pl.position), "color": tuple(pl.color),
                "intensity": pl.intensity, "range": pl.range,
                "type": LIGHT_POINT, "direction": (0.0, -1.0, 0.0),
                "cos_inner": -1.0, "cos_outer": -1.0,
            })
        for obj in self.objects:
            if not obj.visible or obj.light is None:
                continue
            lc = obj.light
            pos = tuple(obj.transform.position)
            reach = max(lc.intensity, 0.5)
            if lc.type == LIGHT_SPOT:
                world_dir = glm.normalize(obj.transform.rotation * SPOT_LOCAL_DIR)
                scale = obj.transform.scale
                spread = max((scale.x + scale.z) * 0.5, 0.05)
                outer = min(math.radians(lc.spot_angle_deg) * spread, math.radians(88.0))
                inner = outer * 0.72
                out.append({
                    "position": pos, "color": tuple(lc.color), "intensity": lc.intensity,
                    "range": reach, "type": LIGHT_SPOT,
                    "direction": (world_dir.x, world_dir.y, world_dir.z),
                    "cos_inner": math.cos(inner), "cos_outer": math.cos(outer),
                })
            else:
                out.append({
                    "position": pos, "color": tuple(lc.color), "intensity": lc.intensity,
                    "range": reach, "type": LIGHT_POINT, "direction": (0.0, -1.0, 0.0),
                    "cos_inner": -1.0, "cos_outer": -1.0,
                })
        return out

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
