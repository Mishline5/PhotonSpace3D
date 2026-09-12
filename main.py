"""Interactive demo: opens the window, builds a small demo scene, runs the main loop.

This is an *example app* built on top of the `engine` package, not part of
the engine itself - it exists to exercise and showcase the engine (scene
setup, camera, the TAB settings panel, hardware-adaptive defaults). A future
"Editing" tool would be another such consumer of `engine`, sitting next to
this file rather than inside the engine package.

Controls: move with the physical W/A/S/D key *positions* - GLFW keycodes are
layout-independent (based on physical position, not the printed keycap), so
on an AZERTY keyboard this is labeled Z/Q/S/D on the keycaps (the standard
French gaming convention) while remaining literally WASD on a QWERTY board,
with no separate code path for either layout. TAB opens the settings panel
(shadows/SSAO/ray tracing/volumetrics, each a 0-3 quality level - see
engine/settings_ui.py); ESC closes the panel if open, else quits.
"""
from __future__ import annotations

import argparse

import glfw

from engine.window import Window
from engine.capture import save_screenshot
from engine.clock import Clock, FrameLimiter
from engine.camera import Camera
from engine.mesh import Mesh
from engine.primitives import make_cube, make_plane, make_uv_sphere, make_cylinder
from engine.scene import Scene, SceneObject, Material, RTPrimitive
from engine.renderer import Renderer
from engine.capabilities import detect as detect_capabilities, describe as describe_capabilities
from engine.settings import preset_for_capabilities
from engine.settings_ui import SettingsUI

CUBE_HALF_EXTENT = 0.5
PLANE_SIZE = 12.0

# Understated, physically-plausible materials throughout - no saturated
# "primary color" surfaces. Distinct shapes are told apart by silhouette and
# roughness/metallic contrast, not by flashy hue.
GROUND_ALBEDO = (0.42, 0.41, 0.39)
CUBE_ALBEDO = (0.30, 0.28, 0.26)      # weathered concrete/stone
SPHERE_ALBEDO = (0.55, 0.55, 0.57)    # brushed metal
CYLINDER_ALBEDO = (0.13, 0.12, 0.12)  # dark matte rubber/resin


def build_scene(ctx) -> Scene:
    scene = Scene()
    cube_mesh = Mesh(ctx, *make_cube(CUBE_HALF_EXTENT))
    plane_mesh = Mesh(ctx, *make_plane(PLANE_SIZE))
    sphere_mesh = Mesh(ctx, *make_uv_sphere(0.45))
    cylinder_mesh = Mesh(ctx, *make_cylinder(0.32, 0.9))

    half = PLANE_SIZE * 0.5
    scene.add(SceneObject(
        "ground", plane_mesh, Material(albedo=GROUND_ALBEDO, metallic=0.0, roughness=0.35),
        rt_primitive=RTPrimitive(kind=0, half_extents=(half, 0.0, half)),
    ))

    cube = scene.add(SceneObject(
        "cube", cube_mesh, Material(albedo=CUBE_ALBEDO, metallic=0.0, roughness=0.65),
        rt_primitive=RTPrimitive(kind=1, half_extents=(CUBE_HALF_EXTENT,) * 3),
    ))
    cube.transform.position = (0.0, 0.5, 0.0)

    sphere = scene.add(SceneObject(
        "sphere", sphere_mesh, Material(albedo=SPHERE_ALBEDO, metallic=0.9, roughness=0.3),
        rt_primitive=RTPrimitive(kind=2, half_extents=(0.45, 0.0, 0.0)),
    ))
    sphere.transform.position = (-1.35, 0.45, 0.7)

    # Cylinders don't have an RT intersection routine yet (see scene.py's
    # RTPrimitive docstring) - a documented limitation, not an omission.
    cylinder = scene.add(SceneObject(
        "cylinder", cylinder_mesh, Material(albedo=CYLINDER_ALBEDO, metallic=0.0, roughness=0.85),
    ))
    cylinder.transform.position = (1.35, 0.45, -0.5)

    return scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=None,
                         help="Exit automatically after N frames (dev/CI use).")
    parser.add_argument("--screenshot", type=str, default=None,
                         help="Save a PNG of the last rendered frame to this path before exiting.")
    parser.add_argument("--shadows", type=int, default=None, help="Override the shadow level (0-3).")
    parser.add_argument("--ssao", type=int, default=None, help="Override the SSAO level (0-3).")
    parser.add_argument("--rt", type=int, default=None, help="Override the ray-tracing level (0-3).")
    parser.add_argument("--volumetric", type=int, default=None, help="Override the volumetric level (0-3).")
    parser.add_argument("--settings-open", action="store_true", help="Start with the settings panel open.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window = Window(args.width, args.height, "3D Engine")
    ctx = window.ctx

    caps = detect_capabilities(ctx)
    print(f"Detected GPU: {describe_capabilities(caps)}")
    settings = preset_for_capabilities(caps)
    if args.shadows is not None:
        settings.shadow_level = args.shadows
    if args.ssao is not None:
        settings.ssao_level = args.ssao
    if args.rt is not None:
        settings.rt_level = args.rt
    if args.volumetric is not None:
        settings.volumetric_level = args.volumetric

    scene = build_scene(ctx)
    camera = Camera()
    camera.set_aspect(*window.framebuffer_size)
    renderer = Renderer(ctx, window.framebuffer_size, settings)
    window.set_vsync(settings.vsync)

    settings_ui = SettingsUI(window)
    settings_ui.visible = args.settings_open

    clock = Clock()
    limiter = FrameLimiter(target_fps=settings.target_fps)
    if args.frames is None and not settings_ui.visible:
        window.set_cursor_captured(True)

    fps_accum = 0.0
    fps_frames = 0
    frame_index = 0

    while not window.should_close():
        limiter.begin_frame()
        dt = clock.tick()
        window.poll_events()

        if window.input.consume_pressed(glfw.KEY_ESCAPE):
            if settings_ui.visible:
                settings_ui.visible = False
                if args.frames is None:
                    window.set_cursor_captured(True)
            else:
                break

        if window.input.consume_pressed(glfw.KEY_TAB):
            settings_ui.toggle()
            if args.frames is None:
                window.set_cursor_captured(not settings_ui.visible)

        new_size = window.poll_resize()
        if new_size is not None:
            camera.set_aspect(*new_size)
            renderer.resize(new_size)

        # Gameplay input is suspended while the settings panel is open -
        # otherwise dragging a slider would also spin the camera and WASD
        # would fly it away underneath the panel.
        if not settings_ui.visible:
            dx, dy = window.input.consume_mouse_delta()
            camera.look(dx, dy)

            forward = float(window.input.is_down(glfw.KEY_W)) - float(window.input.is_down(glfw.KEY_S))
            strafe = float(window.input.is_down(glfw.KEY_D)) - float(window.input.is_down(glfw.KEY_A))
            vertical = float(window.input.is_down(glfw.KEY_SPACE)) - float(window.input.is_down(glfw.KEY_LEFT_CONTROL))
            sprint = window.input.is_down(glfw.KEY_LEFT_SHIFT)
            camera.move(dt, forward, strafe, vertical, sprint)
        else:
            window.input.consume_mouse_delta()

        limiter.target_fps = None if settings.vsync else settings.target_fps
        renderer.render(scene, camera, window.time(), dt)
        settings_ui.draw(window, dt, settings, caps)
        settings_ui.render()

        frame_index += 1
        if args.screenshot and (args.frames is None or frame_index == args.frames):
            save_screenshot(ctx, args.screenshot, window.framebuffer_size)

        window.swap_buffers()
        window.input.end_frame()

        fps_accum += dt
        fps_frames += 1
        if fps_accum >= 0.5:
            window.set_title(f"3D Engine | {fps_frames / fps_accum:.0f} FPS | dt={dt * 1000:.2f}ms")
            fps_accum = 0.0
            fps_frames = 0

        if args.frames is not None and frame_index >= args.frames:
            break

        limiter.wait()

    window.close()


if __name__ == "__main__":
    main()
