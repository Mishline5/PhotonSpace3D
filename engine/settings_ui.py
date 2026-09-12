"""In-engine settings panel (TAB to open): shadows/SSAO/RT/volumetrics as
0-3 level sliders (0 = off), plus exposure and vsync.

Built on Dear ImGui via `pyimgui` (PyPI package `imgui`, not the newer
`imgui_bundle` - the latter bundles its own copy of GLFW as a native
library, which loaded *alongside* the `glfw` package already used for the
window/context and tripped macOS's Objective-C runtime into a duplicate-
class-definition warning, a real reliability risk, not just cosmetic noise.
`pyimgui` has no such bundled dependency).

Deliberately does NOT let ImGui attach its own GLFW callbacks
(`attach_callbacks=False`): this engine's own `Window`/`Input` stay the
single source of truth for input, and this module just reads from them each
frame and forwards what ImGui needs (mouse position/buttons/scroll). This
also means the panel is reusable as-is by the future Editing UI without two
separate input systems fighting over the same window's callbacks.
"""
from __future__ import annotations

import imgui
from imgui.integrations.glfw import GlfwRenderer

from .window import Window
from .settings import QualitySettings, MAX_LEVEL, preset_for_capabilities
from .capabilities import GraphicsCapabilities

_LEVEL_LABELS = {0: "Off", 1: "Low", 2: "Medium", 3: "High"}


def _level_slider(label: str, value: int, disabled_hint: str | None = None) -> int:
    imgui.push_item_width(160)
    changed, value = imgui.slider_int(label, value, 0, MAX_LEVEL, format=f"{_LEVEL_LABELS.get(value, '?')} (%d)")
    imgui.pop_item_width()
    if disabled_hint:
        imgui.same_line()
        imgui.text_disabled(f"({disabled_hint})")
    return value


class SettingsUI:
    def __init__(self, window: Window) -> None:
        imgui.create_context()
        self.io = imgui.get_io()
        self.io.display_size = window.framebuffer_size
        self.io.ini_file_name = None  # a demo/engine sample has no persisted window layout to remember
        self.renderer = GlfwRenderer(window.handle, attach_callbacks=False)
        self.visible = False

    def toggle(self) -> None:
        self.visible = not self.visible

    def _sync_input(self, window: Window, dt: float) -> None:
        io = self.io
        io.display_size = window.framebuffer_size
        io.delta_time = max(dt, 1.0 / 1000.0)
        io.mouse_pos = window.input.mouse_pos
        io.mouse_down[0] = 0 in window.input.mouse_buttons_down
        io.mouse_down[1] = 1 in window.input.mouse_buttons_down
        io.mouse_down[2] = 2 in window.input.mouse_buttons_down
        io.mouse_wheel = window.input.consume_scroll_delta()

    def draw(self, window: Window, dt: float, settings: QualitySettings, caps: GraphicsCapabilities) -> None:
        """No-op (and no GL work at all) when the panel is closed."""
        if not self.visible:
            return

        self._sync_input(window, dt)
        imgui.new_frame()

        imgui.set_next_window_size(420, 0, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(20, 20, condition=imgui.FIRST_USE_EVER)
        imgui.begin("Settings (TAB to close)")

        imgui.text(f"GPU: {caps.renderer}")
        imgui.text_disabled(f"Vendor: {caps.vendor}  |  {caps.gl_version}  |  detected tier: {caps.tier}")
        if caps.is_apple_silicon:
            imgui.text_disabled("Apple GPU: capped at OpenGL 4.1 by the OS, no compute-shader RT path.")
        imgui.separator()

        imgui.text("Quality (0 = off)")
        settings.shadow_level = _level_slider("Shadows", settings.shadow_level)
        settings.ssao_level = _level_slider("Ambient occlusion", settings.ssao_level)
        settings.rt_level = _level_slider("Ray-traced reflections", settings.rt_level)
        vol_hint = "needs shadows" if settings.shadow_level == 0 else None
        settings.volumetric_level = _level_slider("Volumetric light", settings.volumetric_level, vol_hint)

        imgui.separator()
        imgui.text("Display")
        _, settings.exposure = imgui.slider_float("Exposure", settings.exposure, 0.1, 4.0)
        changed, vsync = imgui.checkbox("V-Sync", settings.vsync)
        if changed:
            settings.vsync = vsync
            window.set_vsync(vsync)
        _, settings.target_fps = imgui.slider_float("FPS cap (when V-Sync is off)", settings.target_fps, 15.0, 240.0)

        imgui.separator()
        if imgui.button("Reset to detected hardware default"):
            detected = preset_for_capabilities(caps)
            settings.shadow_level = detected.shadow_level
            settings.ssao_level = detected.ssao_level
            settings.rt_level = detected.rt_level
            settings.volumetric_level = detected.volumetric_level

        imgui.end()
        imgui.render()

    def render(self) -> None:
        if not self.visible:
            return
        self.renderer.render(imgui.get_draw_data())

    @property
    def wants_mouse(self) -> bool:
        return self.visible and self.io.want_capture_mouse

    @property
    def wants_keyboard(self) -> bool:
        return self.visible and self.io.want_capture_keyboard
