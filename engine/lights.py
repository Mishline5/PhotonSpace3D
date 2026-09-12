"""Dynamic light list UBO: an array of point and spot lights the forward
pass shades against, separate from the single sun (directional) light which
stays in the FrameUBO.

Modelled on rt_primitives.RTPrimitivesUBO: its own dedicated GPU storage
(binding 2) re-uploaded every frame from Scene.collect_lights(). Kept out of
the FrameUBO deliberately - the FrameUBO's std140 block is shared, member for
member, with prepass.vert/volumetric.frag, so growing it with a variable-
length light array would be fragile; a separate block that only forward.frag
declares is clean.
"""
from __future__ import annotations

import numpy as np
import moderngl

BINDING = 2
MAX_LIGHTS = 16
# std140: 4 vec4 per light = 64 bytes; ivec4 header = 16 bytes.
LIGHT_STRIDE = 64
HEADER_SIZE = 16
SIZE = HEADER_SIZE + MAX_LIGHTS * LIGHT_STRIDE

TYPE_POINT = 0
TYPE_SPOT = 1


class LightsUBO:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.buffer = ctx.buffer(reserve=SIZE)
        self.buffer.bind_to_uniform_block(BINDING)

    @staticmethod
    def bind_to_program(program: moderngl.Program) -> None:
        try:
            program["Lights"].binding = BINDING
        except KeyError:
            pass

    def update(self, lights: list[dict]) -> None:
        """`lights` is Scene.collect_lights() output: dicts with position,
        color, intensity, range, type, direction, cos_inner, cos_outer."""
        lights = lights[:MAX_LIGHTS]
        header = np.array([len(lights), 0, 0, 0], dtype=np.int32).tobytes()
        body = bytearray()
        for lt in lights:
            px, py, pz = lt["position"]
            r, g, b = lt["color"]
            dx, dy, dz = lt["direction"]
            body += np.array([px, py, pz, lt["range"]], dtype=np.float32).tobytes()
            body += np.array([r, g, b, lt["intensity"]], dtype=np.float32).tobytes()
            body += np.array([dx, dy, dz, float(lt["type"])], dtype=np.float32).tobytes()
            body += np.array([lt["cos_inner"], lt["cos_outer"], 0.0, 0.0], dtype=np.float32).tobytes()
        body += bytes(LIGHT_STRIDE * (MAX_LIGHTS - len(lights)))
        data = header + bytes(body)
        assert len(data) == SIZE, f"Lights payload {len(data)} != {SIZE}"
        self.buffer.write(data)
