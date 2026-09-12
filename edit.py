"""Interactive editing tool: Blender-like selection, transform gizmos
(G/R/S to switch translate/rotate/scale, matching Blender's own keys), and
material editing, on top of the same engine main.py uses.

This is the second consumer of `engine/` design_report.md's original
architecture section anticipated - main.py stays the minimal "engine only"
demo; this is where the graphics Quality panel and everything editing-
related now lives (see editing/ui.py).

Controls: WASD (ZQSD on AZERTY, see main.py's docstring for why - identical
here) to fly, hold the *right* mouse button to look around (unlike main.py,
the cursor here is never captured - see editing/session.py for why: a
precise click needs a free, visible cursor). Left-click an object to select
it, drag one of its gizmo handles to move/rotate/scale it. G/R/S switch the
gizmo mode. TAB opens/closes the Quality panel; ESC closes it if open, else
quits.
"""
from __future__ import annotations

import argparse

import glfw

from engine.window import Window
from engine.capture import save_screenshot
from engine.clock import Clock, FrameLimiter
from engine.camera import Camera
from engine.renderer import Renderer
from engine.capabilities import detect as detect_capabilities, describe as describe_capabilities
from engine.settings import preset_for_capabilities

from editing.session import EditingSession
from editing.picking import ensure_pickable

from main import build_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=None,
                         help="Exit automatically after N frames (dev/CI use).")
    parser.add_argument("--screenshot", type=str, default=None,
                         help="Save a PNG of the last rendered frame to this path before exiting.")
    parser.add_argument("--quality-open", action="store_true", help="Start with the Quality panel open.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window = Window(args.width, args.height, "PhotonSpace3D - Editing")
    ctx = window.ctx
    # Never captured (unlike main.py): a gizmo drag or an object pick needs
    # the cursor visible and reporting its true screen position throughout.
    window.set_cursor_captured(False)

    caps = detect_capabilities(ctx)
    print(f"Detected GPU: {describe_capabilities(caps)}")
    settings = preset_for_capabilities(caps)

    scene = build_scene(ctx)
    ensure_pickable(scene)
    camera = Camera()
    camera.set_aspect(*window.framebuffer_size)
    renderer = Renderer(ctx, window.framebuffer_size, settings)
    window.set_vsync(settings.vsync)

    session = EditingSession(ctx, window)
    session.ui.quality_visible = args.quality_open

    clock = Clock()
    limiter = FrameLimiter(target_fps=settings.target_fps)

    fps_accum = 0.0
    fps_frames = 0
    frame_index = 0

    while not window.should_close():
        limiter.begin_frame()
        dt = clock.tick()
        window.poll_events()

        if window.input.consume_pressed(glfw.KEY_ESCAPE):
            if session.ui.quality_visible:
                session.ui.quality_visible = False
            else:
                break

        if window.input.consume_pressed(glfw.KEY_TAB):
            session.ui.toggle_quality()

        new_size = window.poll_resize()
        if new_size is not None:
            camera.set_aspect(*new_size)
            renderer.resize(new_size)

        # Selection/gizmo-drag/mode-switch/camera-look routing (see
        # editing/session.py) - WASD stays unconditional here, same as
        # main.py, except gated on ImGui not currently owning keyboard focus
        # (typing in a Properties drag-float box shouldn't also fly the camera).
        session.handle_input(window, camera, scene)
        if not session.ui.wants_keyboard:
            forward = float(window.input.is_down(glfw.KEY_W)) - float(window.input.is_down(glfw.KEY_S))
            strafe = float(window.input.is_down(glfw.KEY_D)) - float(window.input.is_down(glfw.KEY_A))
            vertical = float(window.input.is_down(glfw.KEY_SPACE)) - float(window.input.is_down(glfw.KEY_LEFT_CONTROL))
            sprint = window.input.is_down(glfw.KEY_LEFT_SHIFT)
            camera.move(dt, forward, strafe, vertical, sprint)

        limiter.target_fps = None if settings.vsync else settings.target_fps
        renderer.render(scene, camera, window.time(), dt)
        session.render_overlay(camera, window.framebuffer_size)
        session.ui.draw(ctx, window, dt, settings, caps, session.selection)
        session.ui.render()

        frame_index += 1
        if args.screenshot and (args.frames is None or frame_index == args.frames):
            save_screenshot(ctx, args.screenshot, window.framebuffer_size)

        window.swap_buffers()
        window.input.end_frame()

        fps_accum += dt
        fps_frames += 1
        if fps_accum >= 0.5:
            window.set_title(f"PhotonSpace3D - Editing | {fps_frames / fps_accum:.0f} FPS | dt={dt * 1000:.2f}ms")
            fps_accum = 0.0
            fps_frames = 0

        if args.frames is not None and frame_index >= args.frames:
            break

        limiter.wait()

    window.close()


if __name__ == "__main__":
    main()
