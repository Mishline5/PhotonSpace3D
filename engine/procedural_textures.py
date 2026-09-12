"""Procedural PBR texture generation: numpy + Pillow only, entirely in
memory - no image files read from or written to disk, no external assets
downloaded. Consistent with this engine's "from scratch" ethos (procedural
primitives in primitives.py, hand-written shaders) extended to surface
detail: every material below is a subtle, desaturated variation on the exact
scalar tones main.py already used (GROUND_ALBEDO/SPHERE_ALBEDO/etc.), not a
new stylistic direction.

All albedo/roughness/normal arrays are produced in the same linear color
space the rest of the engine's HDR pipeline already assumes (see
forward.frag - there is no sRGB decode anywhere in this pipeline). A texture
generated here must therefore be tuned in that same linear space (mean
around the scalar constant it's replacing), not "what looks right on a
gamma-encoded monitor" - and uploaded to GL without any sRGB internal format.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def value_noise_2d(shape: tuple[int, int], cell_size: float, seed: int) -> np.ndarray:
    """Smooth lattice noise: random values on a coarse grid, bilinearly
    upsampled to `shape`. Resized in 32-bit float mode (PIL mode "F"), not
    quantized to 8-bit first, so this stays smooth even after several
    octaves are summed (fractal_value_noise) without banding."""
    h, w = shape
    rng = np.random.default_rng(seed)
    grid_h = max(int(h / cell_size) + 2, 2)
    grid_w = max(int(w / cell_size) + 2, 2)
    lattice = rng.uniform(0.0, 1.0, size=(grid_h, grid_w)).astype(np.float32)
    img = Image.fromarray(lattice, mode="F").resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def fractal_value_noise(shape: tuple[int, int], octaves: int, seed: int,
                         base_cell: float, persistence: float = 0.5) -> np.ndarray:
    """fBm: sum of `octaves` value-noise layers at halving cell size, each
    weighted less than the last - low-frequency shape from the coarse
    octaves, fine grain from the later ones."""
    result = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    cell = base_cell
    for i in range(octaves):
        result += value_noise_2d(shape, max(cell, 2.0), seed + i * 7919) * amplitude
        total_amplitude += amplitude
        amplitude *= persistence
        cell = max(cell / 2.0, 2.0)
    return result / total_amplitude


def worley_noise_2d(shape: tuple[int, int], num_points: int, seed: int) -> np.ndarray:
    """Cellular noise: distance from each pixel to the nearest of
    `num_points` random feature points, normalized to [0, 1]. Distances wrap
    toroidally (each axis independently) so the result tiles seamlessly -
    the ground plane's UVs already repeat 12x (see primitives.make_plane),
    and a non-tiling noise field would show an ugly seam on that surface."""
    h, w = shape
    rng = np.random.default_rng(seed)
    pts_x = rng.uniform(0.0, w, size=num_points).astype(np.float32)
    pts_y = rng.uniform(0.0, h, size=num_points).astype(np.float32)
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    min_dist = np.full((h, w), np.inf, dtype=np.float32)
    for px, py in zip(pts_x, pts_y):
        dx = np.abs(xs - px)
        dx = np.minimum(dx, w - dx)
        dy = np.abs(ys - py)
        dy = np.minimum(dy, h - dy)
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = np.minimum(min_dist, dist)

    max_possible = float(np.sqrt((w * 0.5) ** 2 + (h * 0.5) ** 2))
    return np.clip(min_dist / max_possible, 0.0, 1.0).astype(np.float32)


def _stretch_axis0(arr: np.ndarray, factor: int) -> np.ndarray:
    """Smear `arr` along axis 0 (down-then-up resize) while leaving axis 1
    untouched - the anisotropic-blur trick behind the brushed-metal streaks:
    axis 1 (across the grain) keeps full detail, axis 0 (along the grain)
    gets heavily blurred, producing directional streaks from isotropic
    input noise without a bespoke directional convolution."""
    h, w = arr.shape
    img = Image.fromarray(arr, mode="F")
    small = img.resize((w, max(h // factor, 1)), Image.BILINEAR)
    back = small.resize((w, h), Image.BILINEAR)
    return np.asarray(back, dtype=np.float32)


def height_to_normal(height: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Tangent-space normal map (RGB in [0, 1], decodes to XYZ in [-1, 1] the
    same way forward.frag's normal-mapping term expects: `texture(...).xyz *
    2.0 - 1.0`) from a height field, via wrapped (tileable) central
    differences - matches the toroidal noise above, no seam at the edges."""
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / length, ny / length, nz / length
    normal = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    return normal.astype(np.float32)


def make_concrete_textures(size: int = 512, seed: int = 1) -> dict[str, np.ndarray]:
    """Ground/cube tone (see main.py's GROUND_ALBEDO/CUBE_ALBEDO comments):
    weathered concrete/stone - coarse fBm shape plus finer Worley pitting."""
    base = fractal_value_noise((size, size), octaves=5, seed=seed, base_cell=size / 8)
    pitting = worley_noise_2d((size, size), num_points=max(size // 16, 8), seed=seed + 1)
    variation = base * 0.7 + pitting * 0.3
    variation -= variation.mean()

    tone = np.array([0.40, 0.39, 0.37], dtype=np.float32)
    albedo = np.clip(tone[None, None, :] + variation[..., None] * 0.10, 0.02, 0.98).astype(np.float32)
    roughness = np.clip(0.45 + variation * 0.25, 0.05, 0.95).astype(np.float32)
    normal = height_to_normal(base, strength=0.6)
    return {"albedo": albedo, "roughness": roughness, "normal": normal}


def make_brushed_metal_textures(size: int = 512, seed: int = 2) -> dict[str, np.ndarray]:
    """Sphere tone (SPHERE_ALBEDO): brushed metal - fine noise smeared into
    directional streaks (see _stretch_axis0)."""
    fine = value_noise_2d((size, size), cell_size=2.0, seed=seed)
    streaked = _stretch_axis0(fine, factor=32)
    variation = streaked - streaked.mean()

    tone = np.array([0.55, 0.55, 0.57], dtype=np.float32)
    albedo = np.clip(tone[None, None, :] + variation[..., None] * 0.05, 0.02, 0.98).astype(np.float32)
    roughness = np.clip(0.30 + variation * 0.20, 0.05, 0.95).astype(np.float32)
    metallic = np.full((size, size), 0.9, dtype=np.float32)
    normal = height_to_normal(streaked, strength=0.3)
    return {"albedo": albedo, "roughness": roughness, "metallic": metallic, "normal": normal}


def make_rubber_textures(size: int = 512, seed: int = 3) -> dict[str, np.ndarray]:
    """Cylinder tone (CYLINDER_ALBEDO): dark matte rubber/resin - fine, low-
    contrast fBm grain."""
    grain = fractal_value_noise((size, size), octaves=4, seed=seed, base_cell=size / 32)
    variation = grain - grain.mean()

    tone = np.array([0.13, 0.12, 0.12], dtype=np.float32)
    albedo = np.clip(tone[None, None, :] + variation[..., None] * 0.03, 0.01, 0.5).astype(np.float32)
    roughness = np.clip(0.85 + variation * 0.08, 0.4, 0.98).astype(np.float32)
    normal = height_to_normal(grain, strength=0.4)
    return {"albedo": albedo, "roughness": roughness, "normal": normal}


def make_plaster_textures(size: int = 512, seed: int = 4) -> dict[str, np.ndarray]:
    """New neutral wall tone: pale plaster/render - broad, gentle fBm, no
    fine grain (a trowelled wall reads as smooth from any distance the
    demo's camera gets to)."""
    base = fractal_value_noise((size, size), octaves=5, seed=seed, base_cell=size / 10)
    variation = base - base.mean()

    tone = np.array([0.72, 0.71, 0.68], dtype=np.float32)
    albedo = np.clip(tone[None, None, :] + variation[..., None] * 0.06, 0.05, 0.98).astype(np.float32)
    roughness = np.clip(0.80 + variation * 0.10, 0.3, 0.98).astype(np.float32)
    normal = height_to_normal(base, strength=0.35)
    return {"albedo": albedo, "roughness": roughness, "normal": normal}


MATERIAL_PRESETS = {
    "concrete": make_concrete_textures,
    "brushed_metal": make_brushed_metal_textures,
    "rubber": make_rubber_textures,
    "plaster": make_plaster_textures,
}
