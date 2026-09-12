"""GLFW window/context creation and low-level input state tracking.

This is the only place that creates the OpenGL context. The exact GLFW
window hints below are what make context creation succeed on macOS at all:
Apple's driver rejects the request outright (glfwCreateWindow returns NULL)
unless FORWARD_COMPAT is set alongside an explicit 4.1 core request - it is
not a soft fallback, it is a hard failure to open a window without it. The
same hints are also correct (and harmless) on Linux/Windows.
"""
from __future__ import annotations

import glfw
import moderngl


class Input:
    """Tracks per-frame keyboard/mouse state from GLFW callbacks.

    Distinguishes "held" (is_down) from "just pressed this frame"
    (consume_pressed): camera movement needs the former, effect toggles need
    the latter - a toggle read with is_down would flip every frame the key
    stays held instead of once per press.
    """

    def __init__(self) -> None:
        self._down: set[int] = set()
        self._pressed_this_frame: set[int] = set()
        self.mouse_pos = (0.0, 0.0)
        self.mouse_delta = (0.0, 0.0)
        self._first_mouse = True
        self.mouse_buttons_down: set[int] = set()
        self.scroll_delta = 0.0
        # GLFW reports the cursor in window points; the rest of the engine
        # (ImGui display_size, picking viewport) works in framebuffer pixels,
        # which are 2x on a Retina display. Scale here so mouse_pos/mouse_delta
        # are always in framebuffer pixels - a no-op (1.0) on non-HiDPI.
        self.cursor_scale = (1.0, 1.0)

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS:
            self._down.add(key)
            self._pressed_this_frame.add(key)
        elif action == glfw.RELEASE:
            self._down.discard(key)

    def _on_cursor_pos(self, window, x, y) -> None:
        x *= self.cursor_scale[0]
        y *= self.cursor_scale[1]
        if self._first_mouse:
            self.mouse_pos = (x, y)
            self._first_mouse = False
        dx = x - self.mouse_pos[0]
        dy = y - self.mouse_pos[1]
        self.mouse_delta = (self.mouse_delta[0] + dx, self.mouse_delta[1] + dy)
        self.mouse_pos = (x, y)

    def _on_mouse_button(self, window, button, action, mods) -> None:
        if action == glfw.PRESS:
            self.mouse_buttons_down.add(button)
        elif action == glfw.RELEASE:
            self.mouse_buttons_down.discard(button)

    def _on_scroll(self, window, xoffset, yoffset) -> None:
        self.scroll_delta += yoffset

    def is_down(self, key: int) -> bool:
        return key in self._down

    def consume_pressed(self, key: int) -> bool:
        """Return True once for a physical key press, then clear it."""
        if key in self._pressed_this_frame:
            self._pressed_this_frame.discard(key)
            return True
        return False

    def consume_mouse_delta(self) -> tuple[float, float]:
        d = self.mouse_delta
        self.mouse_delta = (0.0, 0.0)
        return d

    def consume_scroll_delta(self) -> float:
        d = self.scroll_delta
        self.scroll_delta = 0.0
        return d

    def end_frame(self) -> None:
        self._pressed_this_frame.clear()


class Window:
    def __init__(self, width: int = 1280, height: int = 720, title: str = "3D Engine",
                 fullscreen: bool = False) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 0)  # engine does its own HDR/AO passes, no MSAA
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        monitor = None
        if fullscreen:
            monitor = glfw.get_primary_monitor()
            if monitor:
                mode = glfw.get_video_mode(monitor)
                width, height = mode.size.width, mode.size.height
                glfw.window_hint(glfw.RED_BITS, mode.bits.red)
                glfw.window_hint(glfw.GREEN_BITS, mode.bits.green)
                glfw.window_hint(glfw.BLUE_BITS, mode.bits.blue)
                glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)

        self.handle = glfw.create_window(width, height, title, monitor, None)
        if not self.handle:
            glfw.terminate()
            raise RuntimeError(
                "glfw.create_window failed: no OpenGL 4.1 core context available "
                "on this machine/driver."
            )

        glfw.make_context_current(self.handle)
        self.ctx = moderngl.create_context(require=410)

        self.input = Input()
        glfw.set_key_callback(self.handle, self.input._on_key)
        glfw.set_cursor_pos_callback(self.handle, self.input._on_cursor_pos)
        glfw.set_mouse_button_callback(self.handle, self.input._on_mouse_button)
        glfw.set_scroll_callback(self.handle, self.input._on_scroll)

        self.framebuffer_size = glfw.get_framebuffer_size(self.handle)
        self._resize_pending = False
        glfw.set_framebuffer_size_callback(self.handle, self._on_framebuffer_size)
        self._update_cursor_scale()

        self._vsync = True
        glfw.swap_interval(1)

    def _update_cursor_scale(self) -> None:
        """Ratio of framebuffer pixels to window points (2.0 on Retina, 1.0
        elsewhere) - so Input reports the cursor in framebuffer pixels."""
        ww, wh = glfw.get_window_size(self.handle)
        fw, fh = self.framebuffer_size
        self.input.cursor_scale = (fw / max(ww, 1), fh / max(wh, 1))

    def _on_framebuffer_size(self, window, width, height) -> None:
        # No GL calls in a GLFW callback - just record the new size. The
        # frame loop reallocates once at the top of the next frame, which
        # also collapses a drag's many callback firings into one reallocation.
        self._resize_pending = True
        self.framebuffer_size = (width, height)
        self._update_cursor_scale()

    def poll_resize(self) -> tuple[int, int] | None:
        """Return the new (width, height) once after a resize, else None.

        Guards the zero-size case: a minimized window reports (0, 0), which
        would otherwise try to allocate a zero-sized render target.
        """
        if not self._resize_pending:
            return None
        self._resize_pending = False
        w, h = self.framebuffer_size
        if w == 0 or h == 0:
            return None
        return (w, h)

    def set_vsync(self, enabled: bool) -> None:
        self._vsync = enabled
        glfw.swap_interval(1 if enabled else 0)

    @property
    def vsync(self) -> bool:
        return self._vsync

    def set_title(self, title: str) -> None:
        glfw.set_window_title(self.handle, title)

    def should_close(self) -> bool:
        return glfw.window_should_close(self.handle)

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self.handle)

    @staticmethod
    def poll_events() -> None:
        glfw.poll_events()

    @staticmethod
    def time() -> float:
        return glfw.get_time()

    def set_cursor_captured(self, captured: bool) -> None:
        mode = glfw.CURSOR_DISABLED if captured else glfw.CURSOR_NORMAL
        glfw.set_input_mode(self.handle, glfw.CURSOR, mode)

    def close(self) -> None:
        glfw.destroy_window(self.handle)
        glfw.terminate()
