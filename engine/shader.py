"""Shader source loading + program compilation.

Centralized so every program is built the same way and compile errors
(common when hand-writing GLSL 410 - see design_report.md for the Apple
compiler's stricter-than-usual conformance checking) surface with the
offending file name attached instead of a bare GLSL infolog.
"""
from __future__ import annotations

from pathlib import Path

import moderngl

# Shaders live inside the engine package (not a project-root sibling folder)
# so `engine/` stays a single, self-contained, copy-anywhere directory.
SHADER_DIR = Path(__file__).resolve().parent / "shaders"


def _read(name: str, shader_dir: Path) -> str:
    return (shader_dir / name).read_text()


def load_program(ctx: moderngl.Context, vertex: str, fragment: str,
                  shader_dir: Path | None = None) -> moderngl.Program:
    """`shader_dir` defaults to engine/shaders (SHADER_DIR) - the only
    consumer that ever passes a different one is editing/, loading its own
    gizmo shaders from editing/shaders/, so engine/ stays self-contained
    (nothing inside it ever calls this with a non-default shader_dir)."""
    directory = shader_dir or SHADER_DIR
    try:
        return ctx.program(
            vertex_shader=_read(vertex, directory),
            fragment_shader=_read(fragment, directory),
        )
    except moderngl.Error as exc:
        raise RuntimeError(f"Shader compile/link failed ({vertex} + {fragment}):\n{exc}") from exc
