"""Mesh-accurate collision for the game character.

The character is a sphere; the world is the exact triangles of every
collision-enabled object (see SceneObject.collision_enabled). Colliding a
sphere against real triangles - rather than an AABB - is what makes a rotated
cone read as a ramp you walk up instead of a wall you hit: the sphere slides
along whatever surface it touches, and a surface tilted less than vertical
naturally pushes it up and forward.

Everything is vectorised over triangles with numpy (one batched closest-
point-on-triangle across the whole world per resolve iteration), so even a
few thousand triangles resolve in well under a millisecond.
"""
from __future__ import annotations

import numpy as np
import glm

from engine.scene import Scene
from engine.gl_math import mat4_to_array


def build_world_triangles(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-space triangle soup (A, B, C arrays, each (N,3)) of every visible,
    collision-enabled, non-marker object. Built once when the game starts
    (the scene is static in game mode)."""
    a_list, b_list, c_list = [], [], []
    for obj in scene.iter_visible():
        if obj.marker or not obj.collision_enabled:
            continue
        pos = obj.mesh.local_positions  # (V,3) float32
        idx = obj.mesh.triangle_indices  # (T*3,)
        # mat4_to_array is column-major [16]; reshaped (4,4) it equals M^T, so
        # row-vectors transform as `homog @ mt` (== (M @ v) for column v).
        mt = mat4_to_array(obj.transform.matrix()).reshape(4, 4)
        homog = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=np.float32)], axis=1)
        world = (homog @ mt)[:, :3].astype(np.float32)
        tri = world[idx.reshape(-1, 3)]
        a_list.append(tri[:, 0, :])
        b_list.append(tri[:, 1, :])
        c_list.append(tri[:, 2, :])
    if not a_list:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, empty
    return (np.concatenate(a_list), np.concatenate(b_list), np.concatenate(c_list))


def _closest_on_triangles(p: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Closest point on each triangle to point `p` (Ericson, vectorised over
    triangles). Returns (N,3)."""
    ab = B - A
    ac = C - A
    ap = p[None, :] - A
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p[None, :] - B
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p[None, :] - C
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    result = np.empty_like(A)
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    v = vb / denom
    w = vc / denom
    result[:] = A + ab * v[:, None] + ac * w[:, None]  # default: interior face

    # Edge/vertex regions, applied in priority order (later masks override).
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.clip(np.where((d1 - d3) != 0, d1 / (d1 - d3 + 1e-12), 0.0), 0.0, 1.0)
    result[m] = (A + ab * t[:, None])[m]

    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.clip(np.where((d2 - d6) != 0, d2 / (d2 - d6 + 1e-12), 0.0), 0.0, 1.0)
    result[m] = (A + ac * t[:, None])[m]

    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    denom2 = (d4 - d3) + (d5 - d6)
    t = np.clip(np.where(denom2 != 0, (d4 - d3) / (denom2 + 1e-12), 0.0), 0.0, 1.0)
    result[m] = (B + (C - B) * t[:, None])[m]

    m = (d1 <= 0) & (d2 <= 0)
    result[m] = A[m]
    m = (d3 >= 0) & (d4 <= d3)
    result[m] = B[m]
    m = (d6 >= 0) & (d5 <= d6)
    result[m] = C[m]
    return result


def resolve_sphere(center: glm.vec3, radius: float,
                   tris: tuple[np.ndarray, np.ndarray, np.ndarray],
                   iterations: int = 4) -> tuple[glm.vec3, bool, bool]:
    """Push a sphere out of every triangle it penetrates, iterating so
    stacked contacts settle. Returns (new_center, grounded, hit_ceiling)."""
    A, B, C = tris
    if A.shape[0] == 0:
        return center, False, False
    p = np.array([center.x, center.y, center.z], dtype=np.float32)
    grounded = False
    ceiling = False
    for _ in range(iterations):
        closest = _closest_on_triangles(p, A, B, C)
        delta = p[None, :] - closest
        dist2 = np.einsum("ij,ij->i", delta, delta)
        hit = dist2 < (radius * radius)
        if not np.any(hit):
            break
        # Resolve the single deepest penetration this iteration (stable).
        idx = np.argmin(np.where(hit, dist2, np.inf))
        dist = float(np.sqrt(dist2[idx]))
        if dist < 1e-5:
            continue
        n = delta[idx] / dist
        push = radius - dist
        p = p + n * push
        if n[1] > 0.5:
            grounded = True
        if n[1] < -0.5:
            ceiling = True
    return glm.vec3(float(p[0]), float(p[1]), float(p[2])), grounded, ceiling
