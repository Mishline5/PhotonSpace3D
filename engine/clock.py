"""Delta-time clock and frame-rate limiter.

Two constraints live here: nothing in the engine may move/animate based on
FPS (everything is scaled by measured delta-time), and the engine must never
spin uncapped by default - an unbounded simple scene would otherwise burn
full GPU/CPU power for no visual benefit. The limiter enforces a target
frame time independently of whatever vsync the driver provides, so the cap
still holds even if vsync is off or unavailable.
"""
from __future__ import annotations

import time


class Clock:
    def __init__(self) -> None:
        self._last = time.perf_counter()
        self.delta_time = 0.0
        # Clamp guards against a huge dt after a debugger pause or window
        # drag stall, which would otherwise e.g. teleport the camera through
        # geometry on the next update.
        self.max_delta = 0.25

    def tick(self) -> float:
        now = time.perf_counter()
        self.delta_time = min(now - self._last, self.max_delta)
        self._last = now
        return self.delta_time


class FrameLimiter:
    """Caps the loop to a target FPS via sleep, independent of vsync.

    A plain time.sleep() has ~1-15ms OS scheduler granularity - too coarse to
    hit a frame budget precisely - so this sleeps through all but the last
    slice of the remaining budget, then busy-waits the remainder for
    precision. Pass target_fps=None to disable (uncapped): used only by the
    benchmark tool to measure the engine's FPS ceiling, never the default.
    """

    def __init__(self, target_fps: float | None = 60.0) -> None:
        self.target_fps = target_fps
        self._frame_start = time.perf_counter()
        self._busy_wait_slice = 0.0015

    def begin_frame(self) -> None:
        self._frame_start = time.perf_counter()

    def wait(self) -> None:
        if not self.target_fps:
            return
        target_end = self._frame_start + 1.0 / self.target_fps
        coarse_sleep = target_end - time.perf_counter() - self._busy_wait_slice
        if coarse_sleep > 0:
            time.sleep(coarse_sleep)
        while time.perf_counter() < target_end:
            pass
