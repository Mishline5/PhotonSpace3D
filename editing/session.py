"""Per-frame editor glue: selection, gizmo drag, the transform modes, the
clipboard, sun state, and the click-vs-drag routing for the right mouse
button (a right *click* that didn't move opens the context menu; a right
*drag* orbits the camera). Also the action surface the UI toolbar / context
menu call into (add / delete / copy / paste / set-mode / launch / quit).
"""
from __future__ import annotations

import math

import glfw
import moderngl

from engine.window import Window
from engine.camera import Camera
from engine.scene import Scene

from . import commands
from engine import prefabs
from .selection import Selection
from .gizmo import Gizmo
from .gizmo_interaction import hover_axis, DragState
from .picking import build_pick_ray, pick_object
from .ui import EditingUI

LOOK_MOUSE_BUTTON = 1   # right button
SELECT_MOUSE_BUTTON = 0  # left button
RIGHT_CLICK_MOVE_THRESHOLD = 6.0  # px of travel that turns a right-click into a right-drag (orbit)

MODES = ("translate", "rotate", "scale", "uniform")


class EditingSession:
    def __init__(self, ctx: moderngl.Context, window: Window, scene: Scene, camera: Camera) -> None:
        self.ctx = ctx
        self.window = window
        self.scene = scene
        self.camera = camera
        self.gizmo = Gizmo(ctx)
        self.selection = Selection()
        self.ui = EditingUI(window)
        self.mode = "translate"
        self.hovered_axis: int | None = None
        self._drag: DragState | None = None
        self.clipboard: dict | None = None
        self.launch_requested = False
        self.quit_requested = False

        self._prev_mouse_down: set[int] = set()
        self._right_press_pos: tuple[float, float] | None = None
        self._right_moved = False

        # Sun (directional light) dials the sun panel edits.
        dl = scene.directional_light
        self.sun_azimuth = 210.0
        self.sun_elevation = 48.0
        self.sun_intensity = dl.intensity
        self.sun_color = tuple(dl.color)
        self.apply_sun()

    # --- action surface (called by the UI) --------------------------------
    def set_mode(self, mode: str) -> None:
        if mode in MODES:
            self.mode = mode

    def can_copy(self) -> bool:
        return self.selection.has_selection

    def can_paste(self) -> bool:
        return self.clipboard is not None

    def copy_selected(self) -> None:
        if self.selection.object is not None:
            self.clipboard = commands.snapshot(self.selection.object)

    def paste_clipboard(self) -> None:
        if self.clipboard is not None:
            obj = commands.paste(self.ctx, self.scene, self.clipboard)
            self.selection.set(obj)

    def delete_selected(self) -> None:
        if self.selection.object is not None:
            commands.delete_object(self.scene, self.selection.object)
            self.selection.clear()
            self._drag = None

    def add_object(self, kind: str) -> None:
        if kind in prefabs.LIGHT_KINDS:
            obj = commands.add_light(self.ctx, self.scene, kind, self.camera)
        else:
            obj = commands.add_geometry(self.ctx, self.scene, kind, self.camera)
        self.selection.set(obj)

    def request_launch(self) -> None:
        self.launch_requested = True

    def request_quit(self) -> None:
        self.quit_requested = True

    def apply_sun(self) -> None:
        dl = self.scene.directional_light
        dl.set_from_angles(self.sun_azimuth, self.sun_elevation)
        dl.intensity = self.sun_intensity
        dl.color = self.sun_color

    # --- per-frame input --------------------------------------------------
    def handle_input(self, window: Window, camera: Camera, scene: Scene) -> None:
        viewport_size = window.framebuffer_size
        mouse_pos = window.input.mouse_pos
        inp = window.input
        ui_kbd = self.ui.wants_keyboard
        ui_mouse = self.ui.wants_mouse

        # Transform-mode shortcuts (Blender's G/R/S) + edit shortcuts.
        if not ui_kbd:
            if inp.consume_pressed(glfw.KEY_G):
                self.mode = "translate"
            elif inp.consume_pressed(glfw.KEY_R):
                self.mode = "rotate"
            elif inp.consume_pressed(glfw.KEY_S):
                self.mode = "scale"
            if inp.consume_pressed(glfw.KEY_X) or inp.consume_pressed(glfw.KEY_DELETE):
                self.delete_selected()
            ctrl = inp.is_down(glfw.KEY_LEFT_CONTROL) or inp.is_down(glfw.KEY_LEFT_SUPER)
            if ctrl and inp.consume_pressed(glfw.KEY_C):
                self.copy_selected()
            if ctrl and inp.consume_pressed(glfw.KEY_V):
                self.paste_clipboard()

        down = inp.mouse_buttons_down
        rmb_down = LOOK_MOUSE_BUTTON in down
        rmb_pressed = rmb_down and LOOK_MOUSE_BUTTON not in self._prev_mouse_down
        rmb_released = (not rmb_down) and LOOK_MOUSE_BUTTON in self._prev_mouse_down

        # Right button: a stationary click opens the context menu; a drag orbits.
        if rmb_pressed:
            self._right_press_pos = mouse_pos
            self._right_moved = False
        if rmb_down:
            dx, dy = inp.consume_mouse_delta()
            if self._right_press_pos is not None:
                md = math.hypot(mouse_pos[0] - self._right_press_pos[0],
                                mouse_pos[1] - self._right_press_pos[1])
                if md > RIGHT_CLICK_MOVE_THRESHOLD:
                    self._right_moved = True
            if self._right_moved:
                camera.look(dx, dy)
        else:
            inp.consume_mouse_delta()  # discard so the next orbit doesn't jump
        if rmb_released and not self._right_moved and not ui_mouse:
            self.ui.open_context_menu(mouse_pos)

        # Left button: gizmo drag / selection.
        left_down = SELECT_MOUSE_BUTTON in down
        left_pressed = left_down and SELECT_MOUSE_BUTTON not in self._prev_mouse_down

        if self._drag is not None:
            if left_down:
                self._drag.update(camera, mouse_pos, viewport_size)
            else:
                self._drag = None
        elif not ui_mouse:
            if self.selection.has_selection:
                self.hovered_axis = hover_axis(
                    self.gizmo, camera, self.selection.object, self.mode, mouse_pos, viewport_size)
            else:
                self.hovered_axis = None
            if left_pressed:
                if self.hovered_axis is not None and self.selection.has_selection:
                    self._drag = DragState(self.mode, self.hovered_axis, self.selection.object,
                                           camera, mouse_pos, viewport_size)
                else:
                    ray_origin, ray_dir = build_pick_ray(camera, mouse_pos, viewport_size)
                    self.selection.set(pick_object(scene, ray_origin, ray_dir))

        self._prev_mouse_down = set(down)

    def render_overlay(self, camera: Camera, viewport_size: tuple[int, int]) -> None:
        if not self.selection.has_selection:
            return
        self.gizmo.draw_outline(camera, self.selection.object)
        active_axis = self._drag.axis if self._drag is not None else None
        self.gizmo.draw(camera, self.selection.object, self.mode, self.hovered_axis, active_axis, viewport_size)
