"""Interactive editor: build and dress a scene (add/move/rotate/scale/delete
objects and lights, edit materials, tune graphics, aim the sun), then hit
Launch to walk it in game mode. The second consumer of `engine/` (the first
is main.py); `game/` is the third.

Runs fullscreen by default. Controls: WASD/ZQSD to fly, hold RIGHT mouse to
orbit, a stationary right-click opens the context menu, left-click selects /
drags a gizmo handle. Toolbar (top-centre) has settings, add-object, sun,
the transform modes, launch, and quit. G/R/S switch move/rotate/scale;
Ctrl+C / Ctrl+V copy/paste; X or Delete removes the selection. TAB toggles
the graphics panel; ESC closes an open panel, else quits.
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
from game.session import run_game

from main import build_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--windowed", action="store_true", help="Run in a window instead of fullscreen.")
    parser.add_argument("--frames", type=int, default=None,
                         help="Exit automatically after N frames (dev/CI use; implies windowed).")
    parser.add_argument("--screenshot", type=str, default=None,
                         help="Save a PNG of the last rendered frame to this path before exiting.")
    parser.add_argument("--quality-open", action="store_true", help="Start with the Graphics panel open.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fullscreen = not (args.windowed or args.frames is not None)
    window = Window(args.width, args.height, "PhotonSpace3D - Editing", fullscreen=fullscreen)
    ctx = window.ctx
    # Never captured in the editor: precise clicks need a free, visible cursor.
    window.set_cursor_captured(False)

    caps = detect_capabilities(ctx)
    print(f"Detected GPU: {describe_capabilities(caps)}")
    settings = preset_for_capabilities(caps)
    # Auto-exposure is OFF by default in the editor (on elsewhere) - so a
    # bright light in view doesn't make the whole viewport wash brighter
    # while you're working.
    settings.auto_exposure_level = 0

    scene = build_scene(ctx)
    ensure_pickable(scene)
    camera = Camera()
    camera.set_aspect(*window.framebuffer_size)
    renderer = Renderer(ctx, window.framebuffer_size, settings)
    window.set_vsync(settings.vsync)

    session = EditingSession(ctx, window, scene, camera)
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
            if session.ui.quality_visible or session.ui.sun_visible:
                session.ui.quality_visible = False
                session.ui.sun_visible = False
            else:
                break
        if window.input.consume_pressed(glfw.KEY_TAB):
            session.ui.toggle_quality()

        new_size = window.poll_resize()
        if new_size is not None:
            camera.set_aspect(*new_size)
            renderer.resize(new_size)

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
        session.ui.draw(ctx, window, dt, settings, caps, session)
        session.ui.render()

        frame_index += 1
        if args.screenshot and (args.frames is None or frame_index == args.frames):
            save_screenshot(ctx, args.screenshot, window.framebuffer_size)

        window.swap_buffers()
        window.input.end_frame()

        if session.quit_requested:
            break
        if session.launch_requested:
            session.launch_requested = False
            run_game(window, ctx, renderer, scene, session.ui, caps, settings, camera, clock, limiter)
            # Back in the editor: drop any stale input from the game session.
            window.set_cursor_captured(False)
            window.input.consume_mouse_delta()

        fps_accum += dt
        fps_frames += 1
        if fps_accum >= 0.5:
            window.set_title(f"PhotonSpace3D - Editing | {fps_frames / fps_accum:.0f} FPS")
            fps_accum = 0.0
            fps_frames = 0

        if args.frames is not None and frame_index >= args.frames:
            break

        limiter.wait()

    window.close()


if __name__ == "__main__":
    main()
