"""Explicit, unambiguous glm -> numpy/bytes conversions for UBO uploads.

PyGLM's buffer-protocol/array interop varies across versions and is easy to
get subtly wrong (e.g. silently transposed matrices, which still "render"
but make lighting/camera math wrong in ways that are hard to spot). Every
conversion here is written out element-by-element against GLSL's std140
column-major convention instead, trading a few negligible Python-level ops
(tiny fixed-size vectors/matrices, a handful of times per frame) for
certainty.
"""
from __future__ import annotations

import numpy as np
import glm


def mat4_to_array(m: glm.mat4) -> np.ndarray:
    """Column-major float32[16], matching GLSL's std140 mat4 layout."""
    return np.array([m[c][r] for c in range(4) for r in range(4)], dtype=np.float32)


def vec3_to_array(v, w: float = 0.0) -> np.ndarray:
    """vec3 padded to 4 floats (std140 aligns vec3 as if it were vec4)."""
    return np.array([v[0], v[1], v[2], w], dtype=np.float32)


def mat3_to_array(m: glm.mat3) -> np.ndarray:
    """Column-major float32[9], tightly packed.

    This is the plain-uniform mat3 layout (glUniformMatrix3fv: 3 columns of
    3 floats, no padding) - deliberately NOT std140's rule for a mat3 field
    inside a UBO, which pads every column to 16 bytes. Only use this for
    uniforms declared outside a `layout(std140)` block.
    """
    return np.array([m[c][r] for c in range(3) for r in range(3)], dtype=np.float32)


def normal_matrix(model: glm.mat4) -> glm.mat3:
    """Inverse-transpose of the model's upper 3x3, so normals stay correct
    under non-uniform scale (a plain mat3(model) would skew them)."""
    return glm.transpose(glm.inverse(glm.mat3(model)))
