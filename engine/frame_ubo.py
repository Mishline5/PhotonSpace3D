"""Per-frame uniform block: camera, lights, shadow matrix, and effect toggles.

One small UBO (std140, 368 bytes) written once per frame - this is exactly
the "genuinely dynamic" data the zero-waste-bandwidth constraint expects to
update every frame (camera + lights), as opposed to geometry (uploaded once,
see mesh.py) or the RT primitive list (re-uploaded only when it actually
changes, see rt_primitives.py). Built as an explicit flat byte layout rather
than a numpy structured dtype so the std140 layout is exactly what gets
written, independent of numpy's own native-alignment rules.

GLSL 410 has no `layout(binding=N)` (that needs GL 4.2's
ARB_shading_language_420pack) - so the binding point is set host-side via
`bind_to_program()`, once per program that declares a `Frame` block.
"""
from __future__ import annotations

import numpy as np
import moderngl

from .gl_math import mat4_to_array, vec3_to_array

BINDING = 0
SIZE = 4 * 4 * 4 * 4 + 4 * 4 * 6 + 4 * 4  # 4 mat4 + 6 vec4 + 1 ivec4 = 368 bytes


class FrameUBO:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.buffer = ctx.buffer(reserve=SIZE)
        self.buffer.bind_to_uniform_block(BINDING)

    @staticmethod
    def bind_to_program(program: moderngl.Program) -> None:
        """Point a program's `Frame` uniform block at our binding point."""
        try:
            program["Frame"].binding = BINDING
        except KeyError:
            pass  # program doesn't use the Frame block (e.g. a depth-only pass)

    def write(self, *, view, proj, view_proj, light_view_proj, cam_pos,
              dir_light_dir, dir_light_color, dir_light_intensity,
              point_light_pos, point_light_range, point_light_color, point_light_intensity,
              time_s: float, dt_s: float, screen_w: int, screen_h: int,
              ao_enabled: bool, shadows_enabled: bool, rt_enabled: bool) -> None:
        parts = [
            mat4_to_array(view).tobytes(),
            mat4_to_array(proj).tobytes(),
            mat4_to_array(view_proj).tobytes(),
            mat4_to_array(light_view_proj).tobytes(),
            vec3_to_array(cam_pos, 0.0).tobytes(),
            vec3_to_array(dir_light_dir, 0.0).tobytes(),
            np.array([*dir_light_color, dir_light_intensity], dtype=np.float32).tobytes(),
            np.array([*point_light_pos, point_light_range], dtype=np.float32).tobytes(),
            np.array([*point_light_color, point_light_intensity], dtype=np.float32).tobytes(),
            np.array([time_s, dt_s, float(screen_w), float(screen_h)], dtype=np.float32).tobytes(),
            np.array([int(ao_enabled), int(shadows_enabled), int(rt_enabled), 0], dtype=np.int32).tobytes(),
        ]
        data = b"".join(parts)
        assert len(data) == SIZE, f"FrameUBO payload size {len(data)} != declared {SIZE}"
        self.buffer.write(data)
