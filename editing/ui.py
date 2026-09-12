"""Editor UI (Dear ImGui via pyimgui): a thin icon toolbar pinned top-centre,
a right-click context menu, and the Properties / Quality / Sun panels.

One shared ImGui context (created once). attach_callbacks=False so
engine.window stays the single input source. Everything the UI does routes
through the passed-in `session` action surface (see editing/session.py) - the
UI itself holds only view state (which panels are open, the context-menu
position).
"""
from __future__ import annotations

import glm
import imgui
import moderngl
from imgui.integrations.glfw import GlfwRenderer

from engine.window import Window
from engine.settings import QualitySettings, MAX_LEVEL, preset_for_capabilities
from engine.capabilities import GraphicsCapabilities
from engine.material_textures import apply_texture_kind
from engine import prefabs

from . import icons

_LEVEL_LABELS = {0: "Off", 1: "Low", 2: "Medium", 3: "High"}
_TEXTURE_KINDS = ["none", "noise"]

# Add-menu entries: (kind, icon, label).
_ADD_GEOMETRY = [
    ("cube", icons.cube, "Cube"),
    ("sphere", icons.sphere, "Sphere"),
    ("cylinder", icons.cylinder, "Cylinder"),
    ("cone", icons.cone, "Cone"),
    ("plane", icons.plane, "Plane"),
]
_ADD_LIGHTS = [
    ("point_light", icons.bulb, "Point light"),
    ("spot_light", icons.spot, "Spot light"),
]


def _level_slider(label: str, value: int, hint: str | None = None) -> int:
    imgui.push_item_width(150)
    _, value = imgui.slider_int(label, value, 0, MAX_LEVEL, format=f"{_LEVEL_LABELS.get(value, '?')} (%d)")
    imgui.pop_item_width()
    if hint:
        imgui.same_line()
        imgui.text_disabled(f"({hint})")
    return value


def quality_controls(settings: QualitySettings, caps: GraphicsCapabilities, window: Window) -> None:
    """The graphics-quality slider body, shared by the editor Graphics panel
    and the game-mode TAB panel (game gets its own QualitySettings instance,
    so editing these never touches the editor's - see game/session.py)."""
    imgui.text_disabled(f"GPU: {caps.renderer}  |  tier: {caps.tier}")
    imgui.separator()
    imgui.text("Quality (0 = off)")
    settings.shadow_level = _level_slider("Shadows (PCSS)", settings.shadow_level)
    settings.ssao_level = _level_slider("Ambient occlusion", settings.ssao_level)
    settings.rt_level = _level_slider("Ray-traced reflections", settings.rt_level)
    vol_hint = "needs shadows" if settings.shadow_level == 0 else None
    settings.volumetric_level = _level_slider("Volumetric rays", settings.volumetric_level, vol_hint)
    settings.fog_level = _level_slider("Fog", settings.fog_level)
    settings.msaa_level = _level_slider("MSAA", settings.msaa_level)
    settings.auto_exposure_level = _level_slider("Auto-exposure", settings.auto_exposure_level)

    imgui.separator()
    settings.dust_level = _level_slider("Dust particles", settings.dust_level)
    if settings.dust_level > 0:
        _, settings.dust_size = imgui.slider_float("Dust size", settings.dust_size, 0.4, 4.0)
        _, settings.dust_speed = imgui.slider_float("Dust movement", settings.dust_speed, 0.0, 3.0)
        _, settings.dust_opacity = imgui.slider_float("Dust opacity", settings.dust_opacity, 0.05, 1.0)
        _, dcol = imgui.color_edit3("Dust color", *settings.dust_color)
        settings.dust_color = dcol

    imgui.separator()
    imgui.text("Filters")
    _, settings.grayscale = imgui.checkbox("Black & white", settings.grayscale)
    # Bodycam is a game-only filter (see game/session.py) - not offered here.

    imgui.separator()
    imgui.text("Display")
    _, settings.exposure = imgui.slider_float("Exposure", settings.exposure, 0.1, 4.0)
    changed, vsync = imgui.checkbox("V-Sync", settings.vsync)
    if changed:
        settings.vsync = vsync
        window.set_vsync(vsync)
    _, settings.target_fps = imgui.slider_float("FPS cap", settings.target_fps, 15.0, 240.0)

    imgui.separator()
    if imgui.button("Reset to detected hardware default"):
        d = preset_for_capabilities(caps)
        for a in ("shadow_level", "ssao_level", "rt_level", "volumetric_level",
                  "fog_level", "msaa_level", "auto_exposure_level"):
            setattr(settings, a, getattr(d, a))


class EditingUI:
    def __init__(self, window: Window) -> None:
        imgui.create_context()
        self.io = imgui.get_io()
        self.io.display_size = window.framebuffer_size
        self.io.ini_file_name = None
        self.renderer = GlfwRenderer(window.handle, attach_callbacks=False)

        self.quality_visible = False
        self.properties_visible = True
        self.sun_visible = False
        self._want_open_context = False
        self._context_pos = (0.0, 0.0)

    # --- called by the session --------------------------------------------
    def toggle_quality(self) -> None:
        self.quality_visible = not self.quality_visible

    def open_context_menu(self, pos: tuple[float, float]) -> None:
        self._want_open_context = True
        self._context_pos = pos

    # --- input sync -------------------------------------------------------
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
              caps: GraphicsCapabilities, session) -> None:
        self._sync_input(window, dt)
        imgui.new_frame()

        self._draw_top_bar(window, session)
        self._draw_context_menu(session)
        if self.properties_visible:
            self._draw_properties(ctx, session)
        if self.quality_visible:
            self._draw_quality(window, settings, caps)
        if self.sun_visible:
            self._draw_sun(session)

        imgui.render()

    # --- top toolbar ------------------------------------------------------
    def _draw_top_bar(self, window: Window, session) -> None:
        w = window.framebuffer_size[0]
        imgui.set_next_window_position(w * 0.5, 8.0, imgui.ALWAYS, 0.5, 0.0)
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
                 | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE
                 | imgui.WINDOW_NO_SAVED_SETTINGS)
        imgui.begin("##topbar", flags=flags)

        if icons.icon_button("tb_settings", icons.gear, tooltip="Graphics settings",
                             active=self.quality_visible):
            self.quality_visible = not self.quality_visible
        imgui.same_line()
        if icons.icon_button("tb_add", icons.plus, tooltip="Add object"):
            imgui.open_popup("add_menu")
        self._draw_add_menu(session)
        imgui.same_line()
        if icons.icon_button("tb_sun", icons.sun, tooltip="Sun light", active=self.sun_visible):
            self.sun_visible = not self.sun_visible

        imgui.same_line()
        imgui.dummy(14, 1)  # small gap before the mode group
        imgui.same_line()
        if icons.icon_button("tb_move", icons.move, tooltip="Move (G)", active=session.mode == "translate"):
            session.set_mode("translate")
        imgui.same_line()
        if icons.icon_button("tb_rot", icons.rotate, tooltip="Rotate (R)", active=session.mode == "rotate"):
            session.set_mode("rotate")
        imgui.same_line()
        if icons.icon_button("tb_scale", icons.scale, tooltip="Scale (S)", active=session.mode == "scale"):
            session.set_mode("scale")
        imgui.same_line()
        if icons.icon_button("tb_uscale", icons.scale_all, tooltip="Global scale",
                             active=session.mode == "uniform"):
            session.set_mode("uniform")

        imgui.same_line()
        imgui.dummy(14, 1)
        imgui.same_line()
        if icons.icon_button("tb_play", icons.play, tooltip="Launch (play)"):
            session.request_launch()
        imgui.same_line()
        if icons.icon_button("tb_quit", icons.power, tooltip="Quit engine"):
            session.request_quit()

        imgui.end()

    def _add_row(self, key: str, icon_fn, label: str) -> bool:
        x, y = imgui.get_cursor_screen_pos()
        clicked = imgui.selectable("      " + label)[0]
        dl = imgui.get_window_draw_list()
        icon_fn(dl, x - 1, y - 2, 20, icons._col(icons._FG))
        return clicked

    def _draw_add_menu(self, session) -> None:
        if imgui.begin_popup("add_menu"):
            imgui.text_disabled("Geometry")
            for kind, icon_fn, label in _ADD_GEOMETRY:
                if self._add_row("add_" + kind, icon_fn, label):
                    session.add_object(kind)
            imgui.separator()
            imgui.text_disabled("Lights")
            for kind, icon_fn, label in _ADD_LIGHTS:
                if self._add_row("add_" + kind, icon_fn, label):
                    session.add_object(kind)
            imgui.end_popup()

    # --- context menu -----------------------------------------------------
    def _ctx_row(self, key: str, icon_fn, label: str, enabled: bool = True) -> bool:
        x, y = imgui.get_cursor_screen_pos()
        dl = imgui.get_window_draw_list()
        col = icons._col(icons._FG if enabled else icons._FG_DISABLED)
        if enabled:
            clicked = imgui.selectable("      " + label)[0]
        else:
            imgui.text_disabled("      " + label)
            clicked = False
        icon_fn(dl, x - 1, y - 2, 20, col)
        return clicked

    def _draw_context_menu(self, session) -> None:
        if self._want_open_context:
            imgui.set_next_window_position(self._context_pos[0], self._context_pos[1])
            imgui.open_popup("context_menu")
            self._want_open_context = False

        if imgui.begin_popup("context_menu"):
            has_sel = session.selection.has_selection
            if self._ctx_row("ctx_props", icons.panel, "Properties"):
                self.properties_visible = True
            imgui.separator()
            if self._ctx_row("ctx_move", icons.move, "Move", has_sel):
                session.set_mode("translate")
            if self._ctx_row("ctx_rot", icons.rotate, "Rotate", has_sel):
                session.set_mode("rotate")
            if self._ctx_row("ctx_scale", icons.scale, "Scale", has_sel):
                session.set_mode("scale")
            if self._ctx_row("ctx_uscale", icons.scale_all, "Global scale", has_sel):
                session.set_mode("uniform")
            imgui.separator()
            if self._ctx_row("ctx_copy", icons.copy, "Copy", session.can_copy()):
                session.copy_selected()
            if self._ctx_row("ctx_paste", icons.paste, "Paste", session.can_paste()):
                session.paste_clipboard()
            if self._ctx_row("ctx_del", icons.trash, "Delete", has_sel):
                session.delete_selected()
            imgui.end_popup()

    # --- properties panel -------------------------------------------------
    def _color_circle(self, key: str, rgb: tuple[float, float, float]):
        """A simple color circle that opens a picker on click. Returns
        (changed, rgb)."""
        imgui.push_id(key)
        x, y = imgui.get_cursor_screen_pos()
        r = 11.0
        imgui.invisible_button("swatch", r * 2, r * 2)
        clicked = imgui.is_item_clicked()
        dl = imgui.get_window_draw_list()
        dl.add_circle_filled(x + r, y + r, r, imgui.get_color_u32_rgba(rgb[0], rgb[1], rgb[2], 1.0), 28)
        dl.add_circle(x + r, y + r, r, imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.5), 28, 1.5)
        if clicked:
            imgui.open_popup("colpick")
        changed = False
        if imgui.begin_popup("colpick"):
            imgui.push_item_width(180)
            ch, new_rgb = imgui.color_edit3("##pick", rgb[0], rgb[1], rgb[2])
            imgui.pop_item_width()
            if ch:
                rgb = new_rgb
                changed = True
            imgui.end_popup()
        imgui.pop_id()
        return changed, rgb

    def _draw_properties(self, ctx: moderngl.Context, session) -> None:
        imgui.set_next_window_size(320, 0, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(18, 96, condition=imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin("Properties", True)
        if not opened:
            self.properties_visible = False
        if expanded:
            self._properties_body(ctx, session)
        imgui.end()

    def _properties_body(self, ctx: moderngl.Context, session) -> None:
        obj = session.selection.object
        if obj is None:
            imgui.text_disabled("No object selected.")
            imgui.text_disabled("Left-click one in the scene.")
            return

        imgui.text(obj.name)
        imgui.separator()

        is_light = obj.light is not None
        is_spot = is_light and obj.light.type == prefabs.LIGHT_SPOT

        imgui.text("Transform")
        changed, new_pos = imgui.drag_float3("Position", *tuple(obj.transform.position), 0.02)
        if changed:
            obj.transform.position = new_pos
        # Rotation/scale: shown for geometry and for spot lights (aim + field);
        # a point light is a dimensionless point, so only its position matters.
        if (not is_light) or is_spot:
            euler = tuple(glm.degrees(glm.eulerAngles(obj.transform.rotation)))
            changed, new_euler = imgui.drag_float3("Rotation", *euler, 1.0)
            if changed:
                obj.transform.set_euler_degrees(*new_euler)
            changed, new_scale = imgui.drag_float3("Scale", *tuple(obj.transform.scale), 0.01)
            if changed:
                obj.transform.scale = new_scale

        imgui.separator()
        if is_light:
            imgui.text("Light")
            changed, rgb = self._color_circle("light_col", tuple(obj.light.color))
            imgui.same_line()
            imgui.text("Color")
            if changed:
                obj.light.color = rgb
                prefabs.sync_light_marker(obj)
            _, obj.light.intensity = imgui.slider_float("Intensity", obj.light.intensity, 0.5, 40.0)
            if is_spot:
                _, obj.light.spot_angle_deg = imgui.slider_float(
                    "Cone angle", obj.light.spot_angle_deg, 6.0, 80.0)
            return

        imgui.text("Material")
        mat = obj.material
        changed, rgb = self._color_circle("mat_col", tuple(mat.albedo))
        imgui.same_line()
        imgui.text("Color")
        if changed:
            mat.albedo = rgb
            if mat.texture_kind == "noise":
                apply_texture_kind(ctx, mat)
        _, mat.metallic = imgui.slider_float("Metallic", mat.metallic, 0.0, 1.0)
        _, mat.roughness = imgui.slider_float("Roughness", mat.roughness, 0.03, 1.0)
        _, mat.reflectivity = imgui.slider_float("Reflectivity", mat.reflectivity, 0.0, 1.0)

        imgui.separator()
        imgui.text("Texture")
        cur = _TEXTURE_KINDS.index(mat.texture_kind) if mat.texture_kind in _TEXTURE_KINDS else 0
        changed, idx = imgui.combo("Kind", cur, ["None", "Noise"])
        if changed:
            mat.texture_kind = _TEXTURE_KINDS[idx]
            apply_texture_kind(ctx, mat)
        if mat.texture_kind == "noise":
            g_changed, mat.noise_grain = imgui.slider_float("Grain", mat.noise_grain, 2.0, 48.0)
            s_changed, mat.noise_strength = imgui.slider_float("Strength", mat.noise_strength, 0.0, 2.0)
            if g_changed or s_changed:
                apply_texture_kind(ctx, mat)

        imgui.separator()
        _, obj.collision_enabled = imgui.checkbox("Collision (game mode)", obj.collision_enabled)

    # --- quality panel ----------------------------------------------------
    def _draw_quality(self, window: Window, settings: QualitySettings, caps: GraphicsCapabilities) -> None:
        imgui.set_next_window_size(430, 0, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(18, 470, condition=imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin("Graphics", True)
        if not opened:
            self.quality_visible = False
        if expanded:
            quality_controls(settings, caps, window)
        imgui.end()

    # --- sun panel --------------------------------------------------------
    def _draw_sun(self, session) -> None:
        imgui.set_next_window_size(320, 0, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(360, 96, condition=imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin("Sun light", True)
        if not opened:
            self.sun_visible = False
        if expanded:
            changed = False
            c, session.sun_azimuth = imgui.slider_float("Rotation", session.sun_azimuth, 0.0, 360.0)
            changed = changed or c
            c, session.sun_elevation = imgui.slider_float("Height", session.sun_elevation, 3.0, 90.0)
            changed = changed or c
            c, session.sun_intensity = imgui.slider_float("Intensity", session.sun_intensity, 0.0, 8.0)
            changed = changed or c
            c, rgb = self._color_circle("sun_col", tuple(session.sun_color))
            imgui.same_line()
            imgui.text("Color")
            if c:
                session.sun_color = rgb
                changed = True
            if changed:
                session.apply_sun()
        imgui.end()

    def render(self) -> None:
        self.renderer.render(imgui.get_draw_data())

    def run_frame(self, window: Window, dt: float, body_fn) -> None:
        """Drive one full ImGui frame with arbitrary `body_fn` content, using
        this same host (context + GL renderer). Used by game mode so its TAB
        panel reuses the one ImGui context (creating a second is invalid)."""
        self._sync_input(window, dt)
        imgui.new_frame()
        body_fn()
        imgui.render()
        self.renderer.render(imgui.get_draw_data())

    @property
    def wants_mouse(self) -> bool:
        return self.io.want_capture_mouse

    @property
    def wants_keyboard(self) -> bool:
        return self.io.want_capture_keyboard
