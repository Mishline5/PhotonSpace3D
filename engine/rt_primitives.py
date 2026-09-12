"""RT primitive list UBO: the small set of analytic shapes the hybrid ray-traced
reflection pass can intersect (see forward.frag's trace_reflection).

This is the one buffer in the engine where "skip the upload if nothing
changed" (constraint 1) is both correct and worth doing: unlike a plain
per-object uniform (shared, clobbered by whichever object drew last - see
renderer.py's docstring for why that one is fine to write unconditionally
instead), this UBO has its own dedicated GPU storage, so re-writing it is
purely a function of whether an RT-eligible object's transform actually
changed since the last frame.
"""
from __future__ import annotations

import numpy as np
import moderngl

from .gl_math import mat4_to_array

BINDING = 1
MAX_PRIMITIVES = 8
PRIMITIVE_STRIDE = 64 + 16 + 16 + 16  # mat4 + 3*vec4 = 112 bytes, std140-safe (all vec4/mat4 fields)
HEADER_SIZE = 16  # one ivec4
SIZE = HEADER_SIZE + MAX_PRIMITIVES * PRIMITIVE_STRIDE


class RTPrimitivesUBO:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.buffer = ctx.buffer(reserve=SIZE)
        self.buffer.bind_to_uniform_block(BINDING)
        self._last_signature = None

    @staticmethod
    def bind_to_program(program: moderngl.Program) -> None:
        try:
            program["Primitives"].binding = BINDING
        except KeyError:
            pass  # program doesn't reference the Primitives block

    def update(self, scene) -> None:
        objs = list(scene.iter_rt_primitives())[:MAX_PRIMITIVES]
        signature = tuple(id(o) for o in objs) + tuple(o.transform.version for o in objs)
        if signature == self._last_signature:
            return
        self._last_signature = signature

        header = np.array([len(objs), 0, 0, 0], dtype=np.int32).tobytes()
        body = bytearray()
        for obj in objs:
            model = obj.transform.matrix()
            rt = obj.rt_primitive
            body += mat4_to_array(model).tobytes()
            body += np.array([*obj.material.albedo, obj.material.metallic], dtype=np.float32).tobytes()
            body += np.array(
                [obj.material.roughness, float(rt.kind), obj.material.reflectivity, 0.0], dtype=np.float32
            ).tobytes()
            body += np.array([*rt.half_extents, 0.0], dtype=np.float32).tobytes()
        body += bytes(PRIMITIVE_STRIDE * (MAX_PRIMITIVES - len(objs)))  # unused slots, never read by the shader

        data = header + bytes(body)
        assert len(data) == SIZE, f"Primitives payload size {len(data)} != declared {SIZE}"
        self.buffer.write(data)
