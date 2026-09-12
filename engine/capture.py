"""Dev-only frame capture: dumps the default framebuffer to a PNG.

Exists because this engine is developed/verified in an environment where the
OS-level screenshot tool cannot be granted Screen Recording permission -
reading pixels back from our own live GL framebuffer needs no such
permission (the app already owns the context), so it works everywhere and
doubles as a quick way to produce before/after images for design_report.md.
Not part of the interactive demo's default path (see main.py's --screenshot
flag): the demo runs normally unless that flag is passed.
"""
from __future__ import annotations

import moderngl
from OpenGL import GL
from PIL import Image


def save_screenshot(ctx: moderngl.Context, path: str, size: tuple[int, int]) -> None:
    width, height = size
    # ModernGL's ctx.screen caches the default framebuffer's size from
    # whenever it first detected it, and never refreshes that cache after a
    # window resize - an explicit viewport here (rather than relying on
    # ctx.screen.read()'s default of that stale cached size) is what keeps
    # this correct after the window has been resized.
    data = ctx.screen.read(viewport=(0, 0, width, height), components=3, alignment=1)

    # ctx.screen.read() leaves a spurious GL_INVALID_OPERATION in its wake
    # on this platform/driver - reproduces with a bare-minimum moderngl
    # context (a cleared window and nothing else), so it's a ModernGL/driver
    # characteristic, not anything specific to this engine's rendering, and
    # the pixel data read back is correct regardless (every screenshot this
    # project has ever produced matches what was on screen). Left
    # unaddressed, the dangling error surfaces later as a raised exception
    # the next time anything using PyOpenGL's checked calls runs - e.g.
    # editing/ui.py's Dear ImGui integration, on whatever frame follows a
    # screenshot in the same process. Draining it here (there can be more
    # than one queued) keeps it from ever leaking past this function.
    while GL.glGetError() != GL.GL_NO_ERROR:
        pass

    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # GL reads bottom-to-top
    image.save(path)
