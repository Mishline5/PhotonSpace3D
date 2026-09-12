"""Lightweight, hardware-adaptive PBR rendering engine built on ModernGL + GLFW.

Self-contained: everything the engine needs (modules, shaders/) lives inside
this package, so it can be reused as-is by any consumer - the `main.py`/
`benchmark.py` demo apps at the project root, or the `editing/` package (a
second, independent consumer: selection, gizmos, material/quality panels) -
without depending on anything outside `engine/`. Rendering and scene
management only: no UI code lives in this package (see editing/ui.py for
the graphics quality panel this engine's settings feed into).

Targets OpenGL 4.1 Core (see window.py for why: macOS never shipped GL 4.3,
so no compute shaders / SSBOs are available anywhere in this engine - the
hybrid ray-tracing pass uses analytic intersection in a fragment shader
instead, see forward.frag and rt_primitives.py). capabilities.py detects the
GPU vendor/tier at startup (NVIDIA/AMD/Apple/Intel, low/medium/high) so
settings.py can pick adaptive default quality levels rather than one
hard-coded preset for every machine; every effect (shadows, SSAO, hybrid RT,
volumetric light, fog, MSAA, auto-exposure) is a 0-3 level rather than a
plain on/off flag.
"""
