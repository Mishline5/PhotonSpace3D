"""Procedural shape generators: the engine's equivalent of Blender's Add Mesh menu.

Each function returns (vertices, indices) in the layout mesh.VERTEX_DTYPE
expects, ready to hand to `Mesh(ctx, *make_xxx())`. Segment counts default
to Blender's own primitive defaults (32-segment sphere/cylinder/cone, 16
rings) - low-poly by construction, not because anything here is
approximated. Spheres and cylinder/cone sides use smooth per-vertex normals
(a sphere's true normal genuinely is its radius direction, no averaging
needed to justify it); cube, plane, and the cylinder/cone caps stay
flat-shaded, matching how these primitives look before an artist explicitly
shades them smooth in a DCC tool.
"""
from __future__ import annotations

import numpy as np

from .mesh import VERTEX_DTYPE


def make_cube(half_extent: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Unit cube (24 verts: one duplicated corner set per face for flat
    per-face normals; 36 indices, 2 triangles/face)."""
    h = half_extent
    faces = [
        ((0.0, 0.0, 1.0), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),   # +Z
        ((0.0, 0.0, -1.0), [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)]),  # -Z
        ((1.0, 0.0, 0.0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)]),   # +X
        ((-1.0, 0.0, 0.0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)]),  # -X
        ((0.0, 1.0, 0.0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)]),   # +Y
        ((0.0, -1.0, 0.0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),  # -Y
    ]
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    vertices = np.zeros(24, dtype=VERTEX_DTYPE)
    indices = np.zeros(36, dtype=np.uint32)
    vi = 0
    ii = 0
    for normal, corners in faces:
        base = vi
        for corner, uv in zip(corners, uvs):
            vertices[vi] = (corner, normal, uv)
            vi += 1
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            indices[ii:ii + 3] = (base + a, base + b, base + c)
            ii += 3
    return vertices, indices


def make_box(half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> tuple[np.ndarray, np.ndarray]:
    """General (non-cubic) box, flat-shaded like make_cube (24 verts, 36
    indices). UVs are scaled by each face's own world-space dimensions (1
    world unit = 1 UV unit - the same tiling convention make_plane already
    uses for the ground), not a fixed 0..1 per face - so a texture applied
    to a large, non-uniformly-sized box (e.g. a wall) repeats at a sane,
    roughly constant density instead of smearing one full texture cycle
    across the whole face regardless of size. Use this - not
    make_cube + Transform.scale - for any box whose surface should show a
    tiled procedural texture at something like its real size; make_cube's
    UVs stay 0..1 per face no matter how a Transform later scales it, since
    scale never touches UV coordinates."""
    hx, hy, hz = half_extents
    faces = [
        ((0.0, 0.0, 1.0), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)], (2 * hx, 2 * hy)),
        ((0.0, 0.0, -1.0), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)], (2 * hx, 2 * hy)),
        ((1.0, 0.0, 0.0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)], (2 * hz, 2 * hy)),
        ((-1.0, 0.0, 0.0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)], (2 * hz, 2 * hy)),
        ((0.0, 1.0, 0.0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)], (2 * hx, 2 * hz)),
        ((0.0, -1.0, 0.0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)], (2 * hx, 2 * hz)),
    ]

    vertices = np.zeros(24, dtype=VERTEX_DTYPE)
    indices = np.zeros(36, dtype=np.uint32)
    vi = 0
    ii = 0
    for normal, corners, (uw, uh) in faces:
        uvs = [(0.0, 0.0), (uw, 0.0), (uw, uh), (0.0, uh)]
        base = vi
        for corner, uv in zip(corners, uvs):
            vertices[vi] = (corner, normal, uv)
            vi += 1
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            indices[ii:ii + 3] = (base + a, base + b, base + c)
            ii += 3
    return vertices, indices


def make_plane(size: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Finite quad in the XZ plane, normal +Y. Rendered without face culling
    (see renderer.py), so winding direction doesn't affect visibility."""
    h = size * 0.5
    vertices = np.array([
        ((-h, 0.0, -h), (0.0, 1.0, 0.0), (0.0, 0.0)),
        ((h, 0.0, -h), (0.0, 1.0, 0.0), (size, 0.0)),
        ((h, 0.0, h), (0.0, 1.0, 0.0), (size, size)),
        ((-h, 0.0, h), (0.0, 1.0, 0.0), (0.0, size)),
    ], dtype=VERTEX_DTYPE)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return vertices, indices


def make_uv_sphere(radius: float = 0.5, segments: int = 32, rings: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """UV sphere, Blender's own default segment/ring counts. Smooth-shaded:
    a sphere's normal truly is its radial direction, so no face-splitting or
    normal-averaging is needed to get correct smooth shading."""
    verts = []
    for ring in range(rings + 1):
        theta = ring * np.pi / rings  # 0 at the top pole, pi at the bottom pole
        y = np.cos(theta)
        r_at_ring = np.sin(theta)
        for seg in range(segments + 1):  # +1: duplicate the seam column for correct UVs
            phi = seg * 2.0 * np.pi / segments
            x = r_at_ring * np.cos(phi)
            z = r_at_ring * np.sin(phi)
            normal = (x, y, z)
            position = (x * radius, y * radius, z * radius)
            uv = (seg / segments, ring / rings)
            verts.append((position, normal, uv))

    vertices = np.array(verts, dtype=VERTEX_DTYPE)
    cols = segments + 1
    indices = []
    for ring in range(rings):
        for seg in range(segments):
            a = ring * cols + seg
            b = a + 1
            c = a + cols
            d = c + 1
            # Degenerate triangles naturally collapse to nothing at the
            # poles (all vertices in that ring share one position) - the
            # standard, simplest way to close a UV sphere's poles.
            indices.extend((a, c, d))
            indices.extend((a, d, b))
    return vertices, np.array(indices, dtype=np.uint32)


def make_cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Capped cylinder, Blender's default 32-segment count. Smooth-shaded
    side (the true normal varies continuously around a cylinder's side),
    flat-shaded caps."""
    half_h = height * 0.5
    angles = [seg * 2.0 * np.pi / segments for seg in range(segments + 1)]  # +1: seam column

    verts = []
    # Side: two rows (bottom, top), smooth radial normals.
    for y, v in ((-half_h, 0.0), (half_h, 1.0)):
        for seg, phi in enumerate(angles):
            x, z = np.cos(phi) * radius, np.sin(phi) * radius
            verts.append(((x, y, z), (np.cos(phi), 0.0, np.sin(phi)), (seg / segments, v)))
    side_count = 2 * (segments + 1)

    def add_cap(y: float, normal_y: float) -> list[int]:
        center_index = len(verts)
        verts.append(((0.0, y, 0.0), (0.0, normal_y, 0.0), (0.5, 0.5)))
        rim_indices = []
        for seg, phi in enumerate(angles[:-1]):
            x, z = np.cos(phi) * radius, np.sin(phi) * radius
            rim_indices.append(len(verts))
            verts.append(((x, y, z), (0.0, normal_y, 0.0), (0.5 + 0.5 * np.cos(phi), 0.5 + 0.5 * np.sin(phi))))
        return [center_index, *rim_indices]

    bottom_cap = add_cap(-half_h, -1.0)
    top_cap = add_cap(half_h, 1.0)

    indices = []
    for seg in range(segments):
        a, b = seg, seg + 1
        c, d = side_count // 2 + seg, side_count // 2 + seg + 1
        indices.extend((a, b, d))
        indices.extend((a, d, c))
    for cap, flip in ((bottom_cap, True), (top_cap, False)):
        center = cap[0]
        rim = cap[1:]
        n = len(rim)
        for i in range(n):
            j = (i + 1) % n
            if flip:
                indices.extend((center, rim[j], rim[i]))
            else:
                indices.extend((center, rim[i], rim[j]))

    return np.array(verts, dtype=VERTEX_DTYPE), np.array(indices, dtype=np.uint32)


def make_cone(radius: float = 0.5, height: float = 1.0, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Cone with a flat base, apex at +height/2, Blender's default 32-segment
    count. Smooth-shaded side using the analytic cone lateral normal."""
    half_h = height * 0.5
    angles = [seg * 2.0 * np.pi / segments for seg in range(segments + 1)]

    verts = []
    base_side_indices = []
    for seg, phi in enumerate(angles):
        x, z = np.cos(phi) * radius, np.sin(phi) * radius
        # Analytic lateral normal of a cone: proportional to
        # (height*cos(phi), radius, height*sin(phi)) - tilts the normal
        # upward from purely horizontal by the cone's own half-angle.
        normal = np.array((height * np.cos(phi), radius, height * np.sin(phi)))
        normal = normal / np.linalg.norm(normal)
        base_side_indices.append(len(verts))
        verts.append(((x, -half_h, z), tuple(normal), (seg / segments, 0.0)))

    # The apex needs one distinct vertex per triangle fan wedge (each has a
    # different smooth normal there) - duplicate it per segment instead of
    # reusing a single shared apex vertex.
    apex_per_seg = []
    for seg, phi in enumerate(angles[:-1]):
        mid_phi = phi + np.pi / segments
        normal = np.array((height * np.cos(mid_phi), radius, height * np.sin(mid_phi)))
        normal = normal / np.linalg.norm(normal)
        apex_per_seg.append(len(verts))
        verts.append(((0.0, half_h, 0.0), tuple(normal), (seg / segments + 0.5 / segments, 1.0)))

    indices = []
    for seg in range(segments):
        indices.extend((base_side_indices[seg], base_side_indices[seg + 1], apex_per_seg[seg]))

    # Flat base cap.
    base_center = len(verts)
    verts.append(((0.0, -half_h, 0.0), (0.0, -1.0, 0.0), (0.5, 0.5)))
    base_rim = []
    for seg, phi in enumerate(angles[:-1]):
        x, z = np.cos(phi) * radius, np.sin(phi) * radius
        base_rim.append(len(verts))
        verts.append(((x, -half_h, z), (0.0, -1.0, 0.0), (0.5 + 0.5 * np.cos(phi), 0.5 + 0.5 * np.sin(phi))))
    n = len(base_rim)
    for i in range(n):
        j = (i + 1) % n
        indices.extend((base_center, base_rim[j], base_rim[i]))

    return np.array(verts, dtype=VERTEX_DTYPE), np.array(indices, dtype=np.uint32)


PRIMITIVE_FACTORIES = {
    "cube": make_cube,
    "box": make_box,
    "plane": make_plane,
    "sphere": make_uv_sphere,
    "cylinder": make_cylinder,
    "cone": make_cone,
}
