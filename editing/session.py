"""Per-frame glue: routes left-click to selection/gizmo-drag, right-click-
held to camera look, everything else untouched.

Cursor must stay free and visible for a precise click, which conflicts with
main.py's always-on, cursor-captured mouse-look - the standard resolution
(Unreal/Unity/Blender all do this for their own free-fly viewport camera) is
to gate mouse-look behind a held button instead of it always being live;
camera.move()/camera.look() themselves are untouched, only *when* look() is
called changes. WASD stays unconditional, driven directly by edit.py, same
as main.py.
"""
from __future__ import annotations

import glfw
import moderngl

from engine.window import Window
from engine.camera import Camera
from engine.scene import Scene

from .selection import Selection
from .gizmo import Gizmo
from .gizmo_interaction import hover_axis, DragState
from .picking import build_pick_ray, pick_object
from .ui import EditingUI

LOOK_MOUSE_BUTTON = 1  # right button
SELECT_MOUSE_BUTTON = 0  # left button


class EditingSession:
    def __init__(self, ctx: moderngl.Context, window: Window) -> None:
        self.ctx = ctx
        self.gizmo = Gizmo(ctx)
        self.selection = Selection()
        self.ui = EditingUI(window)
        self.mode = "translate"
        self.hovered_axis: int | None = None
        self._drag: DragState | None = None
        # Input.consume_pressed only tracks a "just pressed" edge for the
        # keyboard (see window.py's Input._on_key vs. _on_mouse_button) - the
        # left mouse button needs the same edge (down this frame, not last)
        # to tell "start a new drag/pick" apart from "still holding from a
        # previous frame", so this session tracks it itself rather than
        # touching engine/window.py for one extra button.
        self._prev_mouse_down: set[int] = set()

    def _left_just_pressed(self, window: Window) -> bool:
        down = window.input.mouse_buttons_down
        return SELECT_MOUSE_BUTTON in down and SELECT_MOUSE_BUTTON not in self._prev_mouse_down

    def handle_input(self, window: Window, camera: Camera, scene: Scene) -> None:
        viewport_size = window.framebuffer_size
        mouse_pos = window.input.mouse_pos

        if not self.ui.wants_keyboard:
            if window.input.consume_pressed(glfw.KEY_G):
                self.mode = "translate"
            elif window.input.consume_pressed(glfw.KEY_R):
                self.mode = "rotate"
            elif window.input.consume_pressed(glfw.KEY_S):
                self.mode = "scale"

        if LOOK_MOUSE_BUTTON in window.input.mouse_buttons_down:
            dx, dy = window.input.consume_mouse_delta()
            camera.look(dx, dy)
        else:
            window.input.consume_mouse_delta()

        ui_has_mouse = self.ui.wants_mouse
        left_just_pressed = self._left_just_pressed(window)
        left_down = SELECT_MOUSE_BUTTON in window.input.mouse_buttons_down

        if self._drag is not None:
            if left_down:
                self._drag.update(camera, mouse_pos, viewport_size)
            else:
                self._drag = None
        elif not ui_has_mouse:
            if self.selection.has_selection:
                self.hovered_axis = hover_axis(
                    self.gizmo, camera, self.selection.object, self.mode, mouse_pos, viewport_size)
            else:
                self.hovered_axis = None

            if left_just_pressed:
                if self.hovered_axis is not None and self.selection.has_selection:
                    self._drag = DragState(
                        self.mode, self.hovered_axis, self.selection.object, camera, mouse_pos, viewport_size)
                else:
                    ray_origin, ray_dir = build_pick_ray(camera, mouse_pos, viewport_size)
                    self.selection.set(pick_object(scene, ray_origin, ray_dir))

        self._prev_mouse_down = set(window.input.mouse_buttons_down)

    def render_overlay(self, camera: Camera, viewport_size: tuple[int, int]) -> None:
        if not self.selection.has_selection:
            return
        active_axis = self._drag.axis if self._drag is not None else None
        self.gizmo.draw(camera, self.selection.object, self.mode, self.hovered_axis, active_axis, viewport_size)
