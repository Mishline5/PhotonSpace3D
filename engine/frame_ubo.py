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
SIZE = 4 * 4 * 4 * 4 + 4 * 4 * 8 + 4 * 4  # 4 mat4 + 8 vec4 + 1 ivec4 = 400 bytes


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
              dir_light_dir, dir_light_color, dir_light_intensity, dir_light_softness,
              point_light_pos, point_light_range, point_light_color, point_light_intensity,
              ambient_sky_color, ambient_ground_color, ambient_intensity,
              time_s: float, dt_s: float, screen_w: int, screen_h: int,
              ao_enabled: bool, shadows_enabled: bool, rt_enabled: bool) -> None:
        parts = [
            mat4_to_array(view).tobytes(),
            mat4_to_array(proj).tobytes(),
            mat4_to_array(view_proj).tobytes(),
            mat4_to_array(light_view_proj).tobytes(),
            vec3_to_array(cam_pos, 0.0).tobytes(),
            # .w is otherwise-unused padding on a direction vector - reused to
            # carry the light's softness dial, same idiom as color.a/pos.a below.
            vec3_to_array(dir_light_dir, dir_light_softness).tobytes(),
            np.array([*dir_light_color, dir_light_intensity], dtype=np.float32).tobytes(),
            np.array([*point_light_pos, point_light_range], dtype=np.float32).tobytes(),
            np.array([*point_light_color, point_light_intensity], dtype=np.float32).tobytes(),
            np.array([time_s, dt_s, float(screen_w), float(screen_h)], dtype=np.float32).tobytes(),
            np.array([int(ao_enabled), int(shadows_enabled), int(rt_enabled), 0], dtype=np.int32).tobytes(),
            # Appended strictly after u_flags (not interleaved earlier) so that
            # prepass.vert/volumetric.frag, which declare a `Frame` block that
            # ends at u_flags, keep reading correct offsets for every field
            # they actually use - only forward.frag's copy of the struct
            # extends past u_flags to see these two.
            np.array([*ambient_sky_color, ambient_intensity], dtype=np.float32).tobytes(),
            np.array([*ambient_ground_color, 0.0], dtype=np.float32).tobytes(),
        ]
        data = b"".join(parts)
        assert len(data) == SIZE, f"FrameUBO payload size {len(data)} != declared {SIZE}"
        self.buffer.write(data)
