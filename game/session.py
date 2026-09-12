"""Game-mode loop, entered from the editor's Launch button and exited with
ESC (back to the editor). Runs the default FPS character over the authored
scene.

Graphics settings in game are a SEPARATE QualitySettings instance: TAB opens
a panel editing *those*, so tuning graphics in game never changes what the
editor shows (and vice-versa). Auto-exposure is on by default here (it's off
in the editor). Light-source markers are hidden. The ImGui host is shared
with the editor (only one ImGui context may exist), passed in as `ui`.
"""
from __future__ import annotations

from dataclasses import replace

import glfw
import imgui

from engine.window import Window
from engine.camera import Camera
from editing.ui import quality_controls

from .character import FpsCharacter
from .collision import build_world_triangles


def _game_panel(gs, caps, window) -> None:
    imgui.set_next_window_size(430, 0, condition=imgui.FIRST_USE_EVER)
    imgui.set_next_window_position(18, 18, condition=imgui.FIRST_USE_EVER)
    imgui.begin("Game graphics (TAB)")
    imgui.text_disabled("Editing these does not change the editor's settings.")
    imgui.separator()
    # Bodycam is a game-only filter, so it lives here rather than in the
    # shared quality controls.
    _, gs.bodycam = imgui.checkbox("Bodycam", gs.bodycam)
    quality_controls(gs, caps, window)
    imgui.end()


def run_game(window: Window, ctx, renderer, scene, ui, caps, editor_settings,
             editor_camera: Camera, clock, limiter) -> None:
    # Independent settings for game mode; auto-exposure on by default here.
    gs = replace(editor_settings)
    if gs.auto_exposure_level == 0:
        gs.auto_exposure_level = 2
    renderer.settings = gs

    camera = Camera(position=(editor_camera.position.x, editor_camera.position.y, editor_camera.position.z),
                    yaw=editor_camera.yaw, pitch=editor_camera.pitch)
    camera.set_aspect(*window.framebuffer_size)
    character = FpsCharacter((float(editor_camera.position.x),
                              max(float(editor_camera.position.y), 2.0),
                              float(editor_camera.position.z)))
    tris = build_world_triangles(scene)

    window.set_cursor_captured(True)
    window.input.consume_mouse_delta()
    panel_visible = False

    while not window.should_close():
        limiter.begin_frame()
        dt = clock.tick()
        window.poll_events()

        if window.input.consume_pressed(glfw.KEY_ESCAPE):
            break  # return to the editor
        if window.input.consume_pressed(glfw.KEY_TAB):
            panel_visible = not panel_visible
            window.set_cursor_captured(not panel_visible)
            window.input.consume_mouse_delta()

        new_size = window.poll_resize()
        if new_size is not None:
            camera.set_aspect(*new_size)
            renderer.resize(new_size)

        dx, dy = window.input.consume_mouse_delta()
        if not panel_visible:
            camera.look(dx, dy)
            forward = float(window.input.is_down(glfw.KEY_W)) - float(window.input.is_down(glfw.KEY_S))
            strafe = float(window.input.is_down(glfw.KEY_D)) - float(window.input.is_down(glfw.KEY_A))
            jump = window.input.is_down(glfw.KEY_SPACE)
        else:
            forward = strafe = 0.0
            jump = False

        camera.fov_degrees = 85.0 if gs.bodycam else 70.0
        character.update(dt, camera, forward, strafe, jump, tris)

        limiter.target_fps = None if gs.vsync else gs.target_fps
        renderer.render(scene, camera, window.time(), dt, draw_markers=False)
        if panel_visible:
            ui.run_frame(window, dt, lambda: _game_panel(gs, caps, window))

        window.swap_buffers()
        window.input.end_frame()
        limiter.wait()

    # Restore the editor's own settings and free the cursor for editing.
    renderer.settings = editor_settings
    window.set_cursor_captured(False)
    window.input.consume_mouse_delta()
