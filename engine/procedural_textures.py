"""Procedural PBR texture generation: numpy + Pillow only, entirely in
memory - no image files read from or written to disk, no external assets
downloaded. Consistent with this engine's "from scratch" ethos (procedural
primitives in primitives.py, hand-written shaders) extended to surface detail.

All albedo/roughness/normal arrays are produced in the same linear color
space the rest of the engine's HDR pipeline already assumes (see
forward.frag - there is no sRGB decode anywhere in this pipeline).

TILING: every generator here produces a *seamlessly tileable* field. The
value/fBm noise samples a periodic integer lattice with modulo-wrapped cell
indices (see value_noise_2d) so the left edge matches the right and the top
matches the bottom exactly - the previous PIL-resize-of-a-random-lattice
approach did NOT wrap and showed a visible seam wherever a mesh's UVs
repeated (the ground plane tiles its UVs many times), which read as a
"weird repeating" artifact. Worley wraps toroidally; normals use wrapped
central differences. So a texture applied to a large tiled surface has no
seams.
"""
from __future__ import annotations

import numpy as np


def value_noise_2d(shape: tuple[int, int], cells: int, seed: int) -> np.ndarray:
    """Seamlessly tileable smooth value noise: a `cells`x`cells` periodic
    lattice of random values, smoothstep-interpolated up to `shape`. Cell
    indices wrap with modulo, and the sample domain spans exactly [0, cells)
    in each axis, so the result tiles perfectly (edge N matches edge 0)."""
    h, w = shape
    cells = max(int(cells), 1)
    rng = np.random.default_rng(seed)
    grid = rng.uniform(0.0, 1.0, size=(cells, cells)).astype(np.float32)

    ys = np.linspace(0.0, cells, h, endpoint=False)
    xs = np.linspace(0.0, cells, w, endpoint=False)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = (ys - y0).astype(np.float32)
    fx = (xs - x0).astype(np.float32)
    y1 = (y0 + 1) % cells
    x1 = (x0 + 1) % cells
    y0 %= cells
    x0 %= cells

    # Smoothstep fade so cell boundaries are C1-continuous (no lattice creases).
    fy = fy * fy * (3.0 - 2.0 * fy)
    fx = fx * fx * (3.0 - 2.0 * fx)

    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x1)]
    g10 = grid[np.ix_(y1, x0)]
    g11 = grid[np.ix_(y1, x1)]
    fx_row = fx[None, :]
    top = g00 * (1.0 - fx_row) + g01 * fx_row
    bot = g10 * (1.0 - fx_row) + g11 * fx_row
    fy_col = fy[:, None]
    return (top * (1.0 - fy_col) + bot * fy_col).astype(np.float32)


def fractal_value_noise(shape: tuple[int, int], octaves: int, seed: int,
                         base_cells: int, persistence: float = 0.5) -> np.ndarray:
    """fBm: sum of octaves at doubling lattice resolution (base_cells,
    2*base_cells, ...). Every octave uses an integer cell count so every
    octave - and thus the sum - stays seamlessly tileable."""
    result = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    cells = max(int(base_cells), 1)
    for i in range(octaves):
        result += value_noise_2d(shape, cells, seed + i * 7919) * amplitude
        total += amplitude
        amplitude *= persistence
        cells *= 2
    return result / max(total, 1e-6)


def worley_noise_2d(shape: tuple[int, int], num_points: int, seed: int) -> np.ndarray:
    """Cellular noise: distance to nearest of `num_points` feature points,
    normalized to [0, 1]. Distances wrap toroidally so it tiles seamlessly."""
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
        min_dist = np.minimum(min_dist, np.sqrt(dx * dx + dy * dy))

    max_possible = float(np.sqrt((w * 0.5) ** 2 + (h * 0.5) ** 2))
    return np.clip(min_dist / max_possible, 0.0, 1.0).astype(np.float32)


def _stretch_axis0(arr: np.ndarray, factor: int) -> np.ndarray:
    """Smear along axis 0 with a wrapped box average - the anisotropic
    brushed-metal streak trick, kept tileable (np.take with mode='wrap')."""
    h, w = arr.shape
    factor = max(int(factor), 1)
    acc = np.zeros_like(arr)
    for offset in range(-factor, factor + 1):
        idx = (np.arange(h) + offset) % h
        acc += arr[idx]
    return (acc / (2 * factor + 1)).astype(np.float32)


def height_to_normal(height: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Tangent-space normal map (RGB in [0,1] -> XYZ in [-1,1]) from a height
    field via wrapped central differences (tileable, matches the noise)."""
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / length, ny / length, nz / length
    return np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1).astype(np.float32)


def make_noise_textures(base_color: tuple[float, float, float],
                        base_roughness: float = 0.6,
                        base_metallic: float = 0.0,
                        grain: float = 8.0,
                        strength: float = 1.0,
                        seed: int = 1,
                        size: int = 512) -> dict[str, np.ndarray]:
    """Generic adjustable grain, derived from a material's own base color -
    this is the "Noise" texture option the editor exposes on any object.

    grain    -> lattice cell count (higher = finer speckle)
    strength -> 0 (no visible texture) .. ~2 (strong)
    """
    cells = max(int(round(grain)), 2)
    base = fractal_value_noise((size, size), octaves=4, seed=seed, base_cells=cells)
    variation = (base - base.mean()) * strength

    tone = np.array(base_color, dtype=np.float32)
    albedo = np.clip(tone[None, None, :] + variation[..., None] * 0.12, 0.01, 0.99).astype(np.float32)
    roughness = np.clip(base_roughness + variation * 0.22, 0.04, 0.99).astype(np.float32)
    metallic = np.full((size, size), float(np.clip(base_metallic, 0.0, 1.0)), dtype=np.float32)
    normal = height_to_normal(base, strength=0.5 * strength)
    return {"albedo": albedo, "roughness": roughness, "metallic": metallic, "normal": normal}


def make_concrete_textures(size: int = 512, seed: int = 1) -> dict[str, np.ndarray]:
    base = fractal_value_noise((size, size), octaves=5, seed=seed, base_cells=8)
    pitting = worley_noise_2d((size, size), num_points=max(size // 16, 8), seed=seed + 1)
    variation = base * 0.7 + pitting * 0.3
    variation -= variation.mean()

    tone = np.array([0.40, 0.39, 0.37], dtype=np.float32)
    albedo = np.clip(tone[None, None, :] + variation[..., None] * 0.10, 0.02, 0.98).astype(np.float32)
    roughness = np.clip(0.45 + variation * 0.25, 0.05, 0.95).astype(np.float32)
    normal = height_to_normal(base, strength=0.6)
    return {"albedo": albedo, "roughness": roughness, "normal": normal}


def make_brushed_metal_textures(size: int = 512, seed: int = 2) -> dict[str, np.ndarray]:
    fine = value_noise_2d((size, size), cells=max(size // 2, 4), seed=seed)
    streaked = _stretch_axis0(fine, factor=max(size // 16, 1))
    variation = streaked - streaked.mean()

    tone = np.array([0.55, 0.55, 0.57], dtype=np.float32)
    albedo = np.clip(tone[None, None, :] + variation[..., None] * 0.05, 0.02, 0.98).astype(np.float32)
    roughness = np.clip(0.30 + variation * 0.20, 0.05, 0.95).astype(np.float32)
    metallic = np.full((size, size), 0.9, dtype=np.float32)
    normal = height_to_normal(streaked, strength=0.3)
    return {"albedo": albedo, "roughness": roughness, "metallic": metallic, "normal": normal}


def make_rubber_textures(size: int = 512, seed: int = 3) -> dict[str, np.ndarray]:
    grain = fractal_value_noise((size, size), octaves=4, seed=seed, base_cells=32)
    variation = grain - grain.mean()

    tone = np.array([0.13, 0.12, 0.12], dtype=np.float32)
    albedo = np.clip(tone[None, None, :] + variation[..., None] * 0.03, 0.01, 0.5).astype(np.float32)
    roughness = np.clip(0.85 + variation * 0.08, 0.4, 0.98).astype(np.float32)
    normal = height_to_normal(grain, strength=0.4)
    return {"albedo": albedo, "roughness": roughness, "normal": normal}


def make_plaster_textures(size: int = 512, seed: int = 4) -> dict[str, np.ndarray]:
    base = fractal_value_noise((size, size), octaves=5, seed=seed, base_cells=10)
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
