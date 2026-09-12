"""Objective performance benchmark: FPS / CPU ms/frame / GPU ms/frame, before
and after each major feature (shadows, SSAO, hybrid RT, volumetric light),
added cumulatively, each at its highest quality level (see engine/settings.py).

Runs each configuration with the frame limiter disabled and vsync off - the
interactive demo (main.py) always caps by default (constraint: never spin
uncapped for no reason), but measuring the engine's real cost needs to see
the uncapped ceiling a cap would otherwise flatten to ~60 FPS regardless of
scene cost. This script is the one deliberate, documented exception to that
default. It also runs one extra baseline pass WITH the cap on, specifically
to demonstrate that the cap actually holds FPS down without changing
CPU/GPU ms per frame - see the "frame limiter" section of the printed output.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from engine.window import Window
from engine.camera import Camera
from engine.clock import FrameLimiter
from engine.renderer import Renderer
from engine.settings import QualitySettings
from main import build_scene

WARMUP_FRAMES = 30
MEASURE_FRAMES = 240
GPU_TIMER_RING_WARMUP = 4  # engine.gpu_timing.RING_SIZE: first samples read stale/zeroed queries
BENCH_LEVEL = 3  # each feature is measured at its highest quality level


@dataclass
class BenchConfig:
    name: str
    shadows: int
    ssao: int
    rt: int
    volumetric: int


CONFIGS = [
    BenchConfig("baseline (PBR only)", 0, 0, 0, 0),
    BenchConfig("+ shadow mapping", BENCH_LEVEL, 0, 0, 0),
    BenchConfig("+ SSAO", BENCH_LEVEL, BENCH_LEVEL, 0, 0),
    BenchConfig("+ hybrid ray-traced reflections", BENCH_LEVEL, BENCH_LEVEL, BENCH_LEVEL, 0),
    BenchConfig("+ volumetric light shafts", BENCH_LEVEL, BENCH_LEVEL, BENCH_LEVEL, BENCH_LEVEL),
]


def run_config(window: Window, scene, camera: Camera, config: BenchConfig) -> dict:
    # A fresh Renderer per config (rather than toggling one shared instance)
    # guarantees no state (e.g. the RT primitives cache, shadow map content)
    # leaks between measurements. GPU resources from prior configs aren't
    # explicitly released - the process exits right after printing results,
    # so that's a deliberate non-issue here, not an oversight.
    settings = QualitySettings(
        shadow_level=config.shadows, ssao_level=config.ssao,
        rt_level=config.rt, volumetric_level=config.volumetric,
    )
    renderer = Renderer(window.ctx, window.framebuffer_size, settings)

    for _ in range(WARMUP_FRAMES):
        window.poll_events()
        renderer.render(scene, camera, window.time(), 1.0 / 60.0)
        window.swap_buffers()

    cpu_ms_samples = []
    gpu_ms_samples = []
    t_start = time.perf_counter()
    for i in range(MEASURE_FRAMES):
        window.poll_events()
        frame_start = time.perf_counter()
        renderer.render(scene, camera, window.time(), 1.0 / 60.0)
        cpu_ms_samples.append((time.perf_counter() - frame_start) * 1000.0)
        window.swap_buffers()
        if i >= GPU_TIMER_RING_WARMUP:
            gpu_ms_samples.append(renderer.gpu_timer.last_ms)
    elapsed = time.perf_counter() - t_start

    return {
        "fps": MEASURE_FRAMES / elapsed,
        "cpu_ms": sum(cpu_ms_samples) / len(cpu_ms_samples),
        "gpu_ms": sum(gpu_ms_samples) / len(gpu_ms_samples),
    }


def run_capped_demo(window: Window, scene, camera: Camera) -> float:
    """Baseline config, cap re-enabled: shows the limiter actually holds FPS
    down (constraint: never run uncapped for no reason) instead of just
    trusting the code."""
    renderer = Renderer(window.ctx, window.framebuffer_size, QualitySettings())
    limiter = FrameLimiter(target_fps=60.0)
    for _ in range(WARMUP_FRAMES):
        limiter.begin_frame()
        window.poll_events()
        renderer.render(scene, camera, window.time(), 1.0 / 60.0)
        window.swap_buffers()
        limiter.wait()

    frames = 90
    t_start = time.perf_counter()
    for _ in range(frames):
        limiter.begin_frame()
        window.poll_events()
        renderer.render(scene, camera, window.time(), 1.0 / 60.0)
        window.swap_buffers()
        limiter.wait()
    elapsed = time.perf_counter() - t_start
    return frames / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=str, default=None,
                         help="Optional path to also save results as a Markdown table.")
    args = parser.parse_args()

    window = Window(args.width, args.height, "benchmark (uncapped)")
    window.set_vsync(False)  # measurement mode only - see module docstring
    ctx = window.ctx
    scene = build_scene(ctx)
    camera = Camera()
    camera.set_aspect(*window.framebuffer_size)

    print(f"Benchmarking at {args.width}x{args.height} "
          f"(framebuffer {window.framebuffer_size[0]}x{window.framebuffer_size[1]}), "
          f"{MEASURE_FRAMES} frames/config, uncapped.\n")

    header = f"{'Configuration':34s}  {'FPS':>8s}  {'CPU ms/frame':>13s}  {'GPU ms/frame':>13s}"
    print(header)
    print("-" * len(header))
    rows = []
    for config in CONFIGS:
        result = run_config(window, scene, camera, config)
        rows.append((config.name, result))
        print(f"{config.name:34s}  {result['fps']:8.1f}  {result['cpu_ms']:13.3f}  {result['gpu_ms']:13.3f}")

    capped_fps = run_capped_demo(window, scene, camera)
    print(f"\nFrame limiter check: baseline config with the 60 FPS cap re-enabled -> {capped_fps:.1f} FPS "
          f"(vs. {rows[0][1]['fps']:.1f} FPS uncapped). Confirms the cap actually holds FPS down "
          f"rather than trusting the code path unverified.")

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"Benchmark: {args.width}x{args.height} "
                    f"(framebuffer {window.framebuffer_size[0]}x{window.framebuffer_size[1]}), "
                    f"{MEASURE_FRAMES} frames/config, uncapped.\n\n")
            f.write("| Configuration | FPS | CPU ms/frame | GPU ms/frame |\n")
            f.write("|---|---|---|---|\n")
            for name, result in rows:
                f.write(f"| {name} | {result['fps']:.1f} | {result['cpu_ms']:.3f} | {result['gpu_ms']:.3f} |\n")
            f.write(f"\nFrame limiter check: baseline with 60 FPS cap re-enabled -> {capped_fps:.1f} FPS "
                    f"(vs. {rows[0][1]['fps']:.1f} FPS uncapped).\n")
        print(f"\nResults written to {args.output}")

    window.close()


if __name__ == "__main__":
    main()
