"""GPU-side static mesh storage: one immutable VBO+IBO pair per shape.

Geometry is uploaded exactly once and never re-uploaded (constraint: zero
wasted CPU->GPU bandwidth) - only per-object transforms and per-frame
uniforms change after startup. See primitives.py for the shape generators
that produce the (vertices, indices) arrays this class stores.
"""
from __future__ import annotations

import numpy as np
import moderngl

# Interleaved position(3f32) + normal(3f32) + uv(2f32) = 32 bytes/vertex.
# Deliberately not further packed (e.g. half-float uv, octahedral normals):
# at the vertex counts these primitives use, the bytes saved are immaterial
# and the added decode complexity is real - see design_report.md for the
# full rationale. Memory-efficiency effort instead goes into never
# re-uploading static data, indexed rendering, and right-sized offscreen
# render targets.
VERTEX_DTYPE = np.dtype([
    ("position", np.float32, 3),
    ("normal", np.float32, 3),
    ("uv", np.float32, 2),
])

# (attribute name, component count) - used to build each program's VAO
# binding dynamically, since different passes use different subsets (e.g.
# the shadow depth-only pass only needs in_position) and GLSL linkers strip
# whichever attributes a given shader doesn't actually reference.
VERTEX_ATTRIBUTES = (("in_position", 3), ("in_normal", 3), ("in_uv", 2))


def _has_attribute(program: moderngl.Program, name: str) -> bool:
    try:
        return isinstance(program[name], moderngl.Attribute)
    except KeyError:
        return False


class Mesh:
    """One immutable VBO+IBO pair, uploaded once at construction time."""

    def __init__(self, ctx: moderngl.Context, vertices: np.ndarray, indices: np.ndarray) -> None:
        assert vertices.dtype == VERTEX_DTYPE
        self.ctx = ctx
        self.vbo = ctx.buffer(vertices.tobytes())
        index_dtype = np.uint32 if len(vertices) > 0xFFFF else np.uint16
        self.ibo = ctx.buffer(indices.astype(index_dtype).tobytes())
        self._index_element_size = 4 if index_dtype is np.uint32 else 2
        self.index_count = len(indices)
        self._vaos: dict[int, moderngl.VertexArray] = {}

    def vertex_array(self, program: moderngl.Program) -> moderngl.VertexArray:
        """One VAO per distinct shader program, created lazily and cached.

        Binds only the attributes `program` actually kept after linking;
        unused ones are skipped over as raw padding bytes so the interleaved
        layout still lines up.
        """
        key = id(program)
        vao = self._vaos.get(key)
        if vao is None:
            format_tokens = []
            attr_names = []
            for name, n in VERTEX_ATTRIBUTES:
                if _has_attribute(program, name):
                    format_tokens.append(f"{n}f")
                    attr_names.append(name)
                else:
                    format_tokens.append(f"{n * 4}x")
            vao = self.ctx.vertex_array(
                program,
                [(self.vbo, " ".join(format_tokens), *attr_names)],
                self.ibo,
                index_element_size=self._index_element_size,
            )
            self._vaos[key] = vao
        return vao

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        self._vaos.clear()
        self.vbo.release()
        self.ibo.release()
