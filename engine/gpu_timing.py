"""Whole-frame GPU timing via a ring-buffered GL_TIME_ELAPSED query.

Honest limitation worth stating precisely: ModernGL's Query object only
exposes the blocking GL_QUERY_RESULT accessor (`.elapsed`) - there is no
GL_QUERY_RESULT_AVAILABLE poll in its public API, so a *true* non-blocking
readback isn't reachable through it. What this does instead: keep a small
ring of in-flight queries and only ever read the OLDEST one, several frames
after it was recorded. By the time that read happens, the GPU has, in
practice, essentially always already finished that work - so `.elapsed`
returns immediately without actually stalling, even though the call could
technically block if the result weren't ready yet. See design_report.md.

A single query wraps the *whole* frame's GPU-side work rather than one query
per pass: GL_TIME_ELAPSED queries cannot be nested or overlapped (only one
can be active at a time per context), and since this engine's passes run
strictly sequentially within a frame anyway, a single whole-frame query is
both the simplest correct option and sufficient - the benchmark tool
measures a feature's GPU cost by comparing whole-frame GPU time with that
feature toggled on vs. off, not by isolating individual passes.
"""
from __future__ import annotations

from collections import deque

import moderngl

RING_SIZE = 4


class GPUFrameTimer:
    def __init__(self, ctx: moderngl.Context, ring_size: int = RING_SIZE) -> None:
        self.ctx = ctx
        self.ring_size = ring_size
        self._pending: deque = deque()
        self.last_ms: float = 0.0

    def begin(self) -> "_FrameScope":
        return _FrameScope(self)

    def _retire(self, query: moderngl.Query) -> None:
        self._pending.append(query)
        while len(self._pending) > self.ring_size:
            oldest = self._pending.popleft()
            self.last_ms = oldest.elapsed / 1_000_000.0  # ns -> ms


class _FrameScope:
    """`with timer.begin(): <submit a frame's GPU work>`"""

    def __init__(self, timer: GPUFrameTimer) -> None:
        self._timer = timer
        self._query = timer.ctx.query(time=True)

    def __enter__(self) -> "_FrameScope":
        self._query.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        self._query.__exit__(*exc_info)
        self._timer._retire(self._query)
