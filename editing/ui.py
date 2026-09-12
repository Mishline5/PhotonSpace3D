"""Editing tool UI: Quality panel (graphics settings - moved here from
engine/settings_ui.py, which no longer exists: "engine does rendering only"
per the user's own split) + a Properties panel (transform/material for
whatever's currently selected).

One shared Dear ImGui context for both windows (imgui.create_context() only
makes sense called once per application - two independent panel objects
each creating their own would fight over the global ImGui state). Reuses
the exact integration pattern the old settings_ui.py established: pyimgui
(not imgui_bundle - see that file's original docstring for the double-GLFW-
native-library crash risk that ruled it out on macOS), attach_callbacks=False
so engine.window.Window/Input stay the single source of truth for input.
"""
from __future__ import annotations

import glm
import imgui
import moderngl
from imgui.integrations.glfw import GlfwRenderer

from engine.window import Window
from engine.settings import QualitySettings, MAX_LEVEL, preset_for_capabilities
from engine.capabilities import GraphicsCapabilities

from .selection import Selection
from .material_presets import PRESETS, apply_preset

_LEVEL_LABELS = {0: "Off", 1: "Low", 2: "Medium", 3: "High"}


def _level_slider(label: str, value: int, disabled_hint: str | None = None) -> int:
    imgui.push_item_width(160)
    changed, value = imgui.slider_int(label, value, 0, MAX_LEVEL, format=f"{_LEVEL_LABELS.get(value, '?')} (%d)")
    imgui.pop_item_width()
    if disabled_hint:
        imgui.same_line()
        imgui.text_disabled(f"({disabled_hint})")
    return value


class EditingUI:
    def __init__(self, window: Window) -> None:
        imgui.create_context()
        self.io = imgui.get_io()
        self.io.display_size = window.framebuffer_size
        self.io.ini_file_name = None
        self.renderer = GlfwRenderer(window.handle, attach_callbacks=False)
        self.quality_visible = False

    def toggle_quality(self) -> None:
        self.quality_visible = not self.quality_visible

    def _sync_input(self, window: Window, dt: float) -> None:
        io = self.io
        io.display_size = window.framebuffer_size
        io.delta_time = max(dt, 1.0 / 1000.0)
        io.mouse_pos = window.input.mouse_pos
        io.mouse_down[0] = 0 in window.input.mouse_buttons_down
        io.mouse_down[1] = 1 in window.input.mouse_buttons_down
        io.mouse_down[2] = 2 in window.input.mouse_buttons_down
        io.mouse_wheel = window.input.consume_scroll_delta()

    def draw(self, ctx: moderngl.Context, window: Window, dt: float, settings: QualitySettings,
              caps: GraphicsCapabilities, selection: Selection) -> None:
        self._sync_input(window, dt)
        imgui.new_frame()

        if self.quality_visible:
            self._draw_quality(window, settings, caps)
        self._draw_properties(ctx, selection)

        imgui.render()

    def _draw_quality(self, window: Window, settings: QualitySettings, caps: GraphicsCapabilities) -> None:
        imgui.set_next_window_size(420, 0, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(20, 20, condition=imgui.FIRST_USE_EVER)
        imgui.begin("Quality (TAB to close)")

        imgui.text(f"GPU: {caps.renderer}")
        imgui.text_disabled(f"Vendor: {caps.vendor}  |  {caps.gl_version}  |  detected tier: {caps.tier}")
        if caps.is_apple_silicon:
            imgui.text_disabled("Apple GPU: capped at OpenGL 4.1 by the OS, no compute-shader RT path.")
        imgui.separator()

        imgui.text("Quality (0 = off)")
        settings.shadow_level = _level_slider("Shadows (PCSS)", settings.shadow_level)
        settings.ssao_level = _level_slider("Ambient occlusion", settings.ssao_level)
        settings.rt_level = _level_slider("Ray-traced reflections", settings.rt_level)
        vol_hint = "needs shadows" if settings.shadow_level == 0 else None
        settings.volumetric_level = _level_slider("Volumetric light", settings.volumetric_level, vol_hint)
        settings.fog_level = _level_slider("Fog", settings.fog_level)
        settings.msaa_level = _level_slider("MSAA", settings.msaa_level)
        settings.auto_exposure_level = _level_slider("Auto-exposure", settings.auto_exposure_level)

        imgui.separator()
        imgui.text("Display")
        _, settings.exposure = imgui.slider_float("Exposure", settings.exposure, 0.1, 4.0)
        changed, vsync = imgui.checkbox("V-Sync", settings.vsync)
        if changed:
            settings.vsync = vsync
            window.set_vsync(vsync)
        _, settings.target_fps = imgui.slider_float(
            "FPS cap (when V-Sync is off)", settings.target_fps, 15.0, 240.0)

        imgui.separator()
        if imgui.button("Reset to detected hardware default"):
            detected = preset_for_capabilities(caps)
            settings.shadow_level = detected.shadow_level
            settings.ssao_level = detected.ssao_level
            settings.rt_level = detected.rt_level
            settings.volumetric_level = detected.volumetric_level
            settings.fog_level = detected.fog_level
            settings.msaa_level = detected.msaa_level
            settings.auto_exposure_level = detected.auto_exposure_level

        imgui.end()

    def _draw_properties(self, ctx: moderngl.Context, selection: Selection) -> None:
        imgui.set_next_window_size(340, 0, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(20, 460, condition=imgui.FIRST_USE_EVER)
        imgui.begin("Properties")

        obj = selection.object
        if obj is None:
            imgui.text_disabled("No object selected - click one in the scene.")
            imgui.end()
            return

        imgui.text(obj.name)
        imgui.separator()

        imgui.text("Transform")
        changed, new_pos = imgui.drag_float3("Position", *tuple(obj.transform.position), 0.02)
        if changed:
            obj.transform.position = new_pos

        euler = tuple(glm.degrees(glm.eulerAngles(obj.transform.rotation)))
        changed, new_euler = imgui.drag_float3("Rotation", *euler, 1.0)
        if changed:
            obj.transform.set_euler_degrees(*new_euler)

        changed, new_scale = imgui.drag_float3("Scale", *tuple(obj.transform.scale), 0.01)
        if changed:
            obj.transform.scale = new_scale

        imgui.separator()
        imgui.text("Material")
        changed, new_albedo = imgui.color_edit3("Albedo", *obj.material.albedo)
        if changed:
            obj.material.albedo = new_albedo
        _, obj.material.roughness = imgui.slider_float("Roughness", obj.material.roughness, 0.03, 1.0)
        _, obj.material.metallic = imgui.slider_float("Metallic", obj.material.metallic, 0.0, 1.0)
        _, obj.material.reflectivity = imgui.slider_float(
            "Reflectivity", obj.material.reflectivity, 0.0, 1.0)

        imgui.separator()
        imgui.text("Texture preset")
        for label, preset_name in PRESETS:
            if imgui.button(label):
                apply_preset(ctx, obj.material, preset_name)
            imgui.same_line()
        imgui.new_line()

        imgui.end()

    def render(self) -> None:
        self.renderer.render(imgui.get_draw_data())

    @property
    def wants_mouse(self) -> bool:
        return self.io.want_capture_mouse

    @property
    def wants_keyboard(self) -> bool:
        return self.io.want_capture_keyboard
