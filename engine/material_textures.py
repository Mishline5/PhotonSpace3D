"""Per-scalar 1x1 default textures for the forward pass's material sampler
uniforms.

forward.frag always samples a texture for albedo/roughness/metallic/normal -
never branches on "does this material have a real map" (see its own
comment) - so a Material with no map set still needs *some* texture bound.
This module builds and caches one 1x1 texture per distinct (kind, value)
pair the engine actually uses, each exactly encoding the scalar it stands in
for, so "no map" reproduces the pre-texturing scalar-uniform behavior
exactly rather than approximately.

dtype="f4" (not the library's default 8-bit-normalized "f1"): matches the
existing precedent in ssao_pass.py's noise texture (plain np.float32 bytes),
and avoids the small quantization a material's exact roughness/albedo value
would otherwise pick up from an 8-bit round-trip.
"""
from __future__ import annotations

import numpy as np
import moderngl

from .scene import Material
from .procedural_textures import MATERIAL_PRESETS

DEFAULT_NORMAL = (0.5, 0.5, 1.0)  # decodes to tangent-space (0, 0, 1): a true no-op, see forward.frag


class MaterialTextures:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._cache: dict[tuple, moderngl.Texture] = {}
        self.default_normal = self._make_texture(DEFAULT_NORMAL)

    def _make_texture(self, value: tuple[float, ...]) -> moderngl.Texture:
        components = len(value)
        data = np.array(value, dtype=np.float32).reshape(1, 1, components)
        tex = self.ctx.texture((1, 1), components, data.tobytes(), dtype="f4")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def get_or_create(self, kind: str, value) -> moderngl.Texture:
        """`kind` is "albedo" (value: an RGB tuple) or a scalar kind name
        ("roughness"/"metallic", value: a float) - matches how Material
        stores each field."""
        if kind == "albedo":
            key = (kind, tuple(round(float(c), 6) for c in value))
        else:
            key = (kind, round(float(value), 6))

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        tex = self._make_texture(tuple(value) if kind == "albedo" else (float(value),))
        self._cache[key] = tex
        return tex


def upload_map(ctx: moderngl.Context, array: np.ndarray) -> moderngl.Texture:
    """Uploads a procedurally generated map (engine/procedural_textures.py) -
    an (H, W) or (H, W, C) float32 array in [0, 1] - as a real GL texture:
    8-bit normalized (moderngl's default dtype, unlike the 1x1 defaults
    above which use "f4" to avoid quantizing an exact scalar) since this is
    genuine, already-noisy visual detail rather than a value that must
    reproduce exactly - and with mipmaps + repeat wrapping, since these are
    meant to tile across a mesh's UV space (e.g. the ground plane repeats
    its UVs 12x - see primitives.make_plane)."""
    h, w = array.shape[:2]
    components = 1 if array.ndim == 2 else array.shape[2]
    quantized = np.ascontiguousarray((np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    tex = ctx.texture((w, h), components, quantized.tobytes())
    tex.repeat_x = True
    tex.repeat_y = True
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.build_mipmaps()
    return tex


def apply_material_preset(ctx: moderngl.Context, material: Material, preset_name: str,
                           size: int = 512, seed: int | None = None) -> None:
    """Generates and uploads one of procedural_textures.MATERIAL_PRESETS
    onto `material`'s map fields - the one place both main.py's demo scene
    and the editing/ material panel apply a named preset, so the two can
    never drift apart into different-looking "concrete"."""
    factory = MATERIAL_PRESETS[preset_name]
    maps = factory(size=size) if seed is None else factory(size=size, seed=seed)
    material.albedo_map = upload_map(ctx, maps["albedo"])
    material.roughness_map = upload_map(ctx, maps["roughness"])
    if "metallic" in maps:
        material.metallic_map = upload_map(ctx, maps["metallic"])
    material.normal_map = upload_map(ctx, maps["normal"])
