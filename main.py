"""Interactive demo: opens the window, builds a small demo scene, runs the main loop.

This is an *example app* built on top of the `engine` package, not part of
the engine itself - it exists to exercise and showcase the engine (scene
setup, camera, hardware-adaptive defaults) with a bare-bones loop and no
editing UI at all - `engine/` owns rendering and scene management only. See
`edit.py` (same directory) for the interactive Blender-like editing tool
(selection, gizmos, material/quality panels), the second consumer of
`engine/` this project's design report anticipated.

Controls: move with the physical W/A/S/D key *positions* - GLFW keycodes are
layout-independent (based on physical position, not the printed keycap), so
on an AZERTY keyboard this is labeled Z/Q/S/D on the keycaps (the standard
French gaming convention) while remaining literally WASD on a QWERTY board,
with no separate code path for either layout. ESC quits.
"""
from __future__ import annotations

import argparse

import glfw

from engine.window import Window
from engine.capture import save_screenshot
from engine.clock import Clock, FrameLimiter
from engine.camera import Camera
from engine.mesh import Mesh
from engine.primitives import make_cube, make_box, make_plane, make_uv_sphere, make_cylinder, make_cone
from engine.scene import Scene, SceneObject, Material, RTPrimitive
from engine.renderer import Renderer
from engine.capabilities import detect as detect_capabilities, describe as describe_capabilities
from engine.settings import preset_for_capabilities
from engine.material_textures import apply_material_preset

CUBE_HALF_EXTENT = 0.5
PLANE_SIZE = 20.0
WALL_HEIGHT = 5.0
WALL_THICKNESS = 0.4

# Understated, physically-plausible materials throughout - no saturated
# "primary color" surfaces. Distinct shapes are told apart by silhouette and
# roughness/metallic contrast, not by flashy hue. Each scalar tone below is
# still set as a real fallback (used verbatim by the hybrid RT pass's
# scalar-only Primitives UBO, see rt_primitives.py) even though the visible
# surface now comes from a procedural texture (apply_material_preset) layered
# on top for every object - "textured" and "flat-scalar-only" were never an
# either/or in this engine's material model (see engine/scene.py's Material).
GROUND_ALBEDO = (0.42, 0.41, 0.39)
CUBE_ALBEDO = (0.30, 0.28, 0.26)      # weathered concrete/stone
SPHERE_ALBEDO = (0.55, 0.55, 0.57)    # brushed metal
CYLINDER_ALBEDO = (0.13, 0.12, 0.12)  # dark matte rubber/resin
WALL_ALBEDO = (0.72, 0.71, 0.68)      # pale plaster/render
PLATFORM_ALBEDO = (0.42, 0.41, 0.39)  # same concrete family as the ground


def build_scene(ctx) -> Scene:
    scene = Scene()
    cube_mesh = Mesh(ctx, *make_cube(CUBE_HALF_EXTENT))
    plane_mesh = Mesh(ctx, *make_plane(PLANE_SIZE))
    sphere_mesh = Mesh(ctx, *make_uv_sphere(0.45))
    cylinder_mesh = Mesh(ctx, *make_cylinder(0.32, 0.9))
    cone_mesh = Mesh(ctx, *make_cone(0.4, 0.9))

    half = PLANE_SIZE * 0.5
    wall_half_height = WALL_HEIGHT * 0.5
    wall_half_thickness = WALL_THICKNESS * 0.5
    # Both walls span the *full* ground half-width along their long axis
    # (not shortened to leave room for the other wall's thickness) so they
    # overlap slightly at the shared corner rather than leaving a gap a
    # shortened-and-repositioned pair would need exact, easy-to-get-wrong
    # mitering to avoid.
    back_wall_mesh = Mesh(ctx, *make_box((half, wall_half_height, wall_half_thickness)))
    side_wall_mesh = Mesh(ctx, *make_box((wall_half_thickness, wall_half_height, half)))
    platform_mesh = Mesh(ctx, *make_box((1.1, 0.14, 1.1)))

    ground = scene.add(SceneObject(
        "ground", plane_mesh, Material(albedo=GROUND_ALBEDO, metallic=0.0, roughness=0.35),
        rt_primitive=RTPrimitive(kind=0, half_extents=(half, 0.0, half)),
    ))
    apply_material_preset(ctx, ground.material, "concrete", seed=1)

    # Two walls, built directly at their real size via make_box (not a unit
    # cube stretched by Transform.scale - see make_box's docstring for why:
    # make_cube's UVs stay 0..1 per face regardless of scale, which would
    # smear one texture cycle across the whole ~19-unit wall instead of
    # tiling it at a sane density). RTPrimitive.half_extents matches the
    # mesh's own real half-extents since there's no Transform.scale left to
    # divide back out.
    back_wall = scene.add(SceneObject(
        "back_wall", back_wall_mesh, Material(albedo=WALL_ALBEDO, metallic=0.0, roughness=0.88),
        rt_primitive=RTPrimitive(kind=1, half_extents=(half, wall_half_height, wall_half_thickness)),
    ))
    back_wall.transform.position = (0.0, wall_half_height, -(half - wall_half_thickness))
    apply_material_preset(ctx, back_wall.material, "plaster", seed=4)

    side_wall = scene.add(SceneObject(
        "side_wall", side_wall_mesh, Material(albedo=WALL_ALBEDO, metallic=0.0, roughness=0.88),
        rt_primitive=RTPrimitive(kind=1, half_extents=(wall_half_thickness, wall_half_height, half)),
    ))
    side_wall.transform.position = (-(half - wall_half_thickness), wall_half_height, 0.0)
    apply_material_preset(ctx, side_wall.material, "plaster", seed=5)

    cube = scene.add(SceneObject(
        "cube", cube_mesh, Material(albedo=CUBE_ALBEDO, metallic=0.0, roughness=0.65),
        rt_primitive=RTPrimitive(kind=1, half_extents=(CUBE_HALF_EXTENT,) * 3),
    ))
    cube.transform.position = (0.0, 0.5, 0.0)
    apply_material_preset(ctx, cube.material, "concrete", seed=2)

    sphere = scene.add(SceneObject(
        "sphere", sphere_mesh, Material(albedo=SPHERE_ALBEDO, metallic=0.9, roughness=0.3),
        rt_primitive=RTPrimitive(kind=2, half_extents=(0.45, 0.0, 0.0)),
    ))
    sphere.transform.position = (-1.35, 0.45, 0.7)
    apply_material_preset(ctx, sphere.material, "brushed_metal", seed=3)

    cylinder = scene.add(SceneObject(
        "cylinder", cylinder_mesh, Material(albedo=CYLINDER_ALBEDO, metallic=0.0, roughness=0.85),
        rt_primitive=RTPrimitive(kind=3, half_extents=(0.32, 0.45, 0.0)),
    ))
    cylinder.transform.position = (1.35, 0.45, -0.5)
    apply_material_preset(ctx, cylinder.material, "rubber", seed=6)

    # A few more elements, still a small, deliberately curated set (not a
    # full environment overhaul) - a low platform, a short wide "drum"
    # variant of the same cylinder mesh, and a cone (visible, but not yet
    # RT-reflective: analytic cone intersection is deferred, see
    # design_report.md, same category of documented gap the cylinder had
    # before this pass).
    platform = scene.add(SceneObject(
        "platform", platform_mesh, Material(albedo=PLATFORM_ALBEDO, metallic=0.0, roughness=0.55),
        rt_primitive=RTPrimitive(kind=1, half_extents=(1.1, 0.14, 1.1)),
    ))
    platform.transform.position = (3.1, 0.14, 1.6)
    apply_material_preset(ctx, platform.material, "concrete", seed=7)

    drum = scene.add(SceneObject(
        "drum", cylinder_mesh, Material(albedo=CYLINDER_ALBEDO, metallic=0.0, roughness=0.8),
        rt_primitive=RTPrimitive(kind=3, half_extents=(0.32, 0.45, 0.0)),
    ))
    # Uniform X/Z scale (radial), independent Y (axial) scale: keeps the
    # cylinder's radial hit-normal transform exact under non-uniform scale
    # (mat3(model) applied to a radial normal stays radial iff X and Z scale
    # match - see the box comment above for the general version of this).
    drum.transform.scale = (1.5, 0.5, 1.5)
    drum.transform.position = (3.0, 0.45 * 0.5, -1.8)
    apply_material_preset(ctx, drum.material, "rubber", seed=8)

    cone = scene.add(SceneObject(
        "cone", cone_mesh, Material(albedo=PLATFORM_ALBEDO, metallic=0.0, roughness=0.6),
    ))
    cone.transform.position = (-2.4, 0.45, -1.6)
    apply_material_preset(ctx, cone.material, "concrete", seed=9)

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
    parser.add_argument("--fog", type=int, default=None, help="Override the fog level (0-3).")
    parser.add_argument("--msaa", type=int, default=None, help="Override the MSAA level (0-3).")
    parser.add_argument("--auto-exposure", type=int, default=None, help="Override the auto-exposure level (0-3).")
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
    if args.fog is not None:
        settings.fog_level = args.fog
    if args.msaa is not None:
        settings.msaa_level = args.msaa
    if args.auto_exposure is not None:
        settings.auto_exposure_level = args.auto_exposure

    scene = build_scene(ctx)
    camera = Camera()
    camera.set_aspect(*window.framebuffer_size)
    renderer = Renderer(ctx, window.framebuffer_size, settings)
    window.set_vsync(settings.vsync)

    clock = Clock()
    limiter = FrameLimiter(target_fps=settings.target_fps)
    if args.frames is None:
        window.set_cursor_captured(True)

    fps_accum = 0.0
    fps_frames = 0
    frame_index = 0

    while not window.should_close():
        limiter.begin_frame()
        dt = clock.tick()
        window.poll_events()

        if window.input.consume_pressed(glfw.KEY_ESCAPE):
            break

        new_size = window.poll_resize()
        if new_size is not None:
            camera.set_aspect(*new_size)
            renderer.resize(new_size)

        dx, dy = window.input.consume_mouse_delta()
        camera.look(dx, dy)

        forward = float(window.input.is_down(glfw.KEY_W)) - float(window.input.is_down(glfw.KEY_S))
        strafe = float(window.input.is_down(glfw.KEY_D)) - float(window.input.is_down(glfw.KEY_A))
        vertical = float(window.input.is_down(glfw.KEY_SPACE)) - float(window.input.is_down(glfw.KEY_LEFT_CONTROL))
        sprint = window.input.is_down(glfw.KEY_LEFT_SHIFT)
        camera.move(dt, forward, strafe, vertical, sprint)

        limiter.target_fps = None if settings.vsync else settings.target_fps
        renderer.render(scene, camera, window.time(), dt)

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
