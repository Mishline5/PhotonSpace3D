"""Sober vector icons drawn straight into Dear ImGui's draw list - no image
files loaded (consistent with the project's from-scratch, no-external-assets
ethos, exactly like the procedural textures and hand-written shaders).

Each icon function paints into the given draw list within a square box
(x, y, size); `icon_button` wraps one in an invisible button with a hover
background so it behaves like a real toolbar button, and greys the glyph out
when disabled (the "you can't click this yet" affordance the tool asks for,
e.g. Paste with an empty clipboard).
"""
from __future__ import annotations

import math

import imgui

_FG = (0.86, 0.87, 0.90, 1.0)
_FG_DISABLED = (0.42, 0.43, 0.47, 1.0)
_HOVER_BG = (1.0, 1.0, 1.0, 0.10)
_ACTIVE_BG = (1.0, 1.0, 1.0, 0.18)


def _col(rgba):
    return imgui.get_color_u32_rgba(*rgba)


def _line(dl, x0, y0, s, a, b, col, thick=1.6):
    dl.add_line(x0 + a[0] * s, y0 + a[1] * s, x0 + b[0] * s, y0 + b[1] * s, col, thick)


# --- individual glyphs: all coordinates are fractions of the box size ------

def gear(dl, x, y, s, col):
    cx, cy = x + s * 0.5, y + s * 0.5
    r = s * 0.30
    for i in range(8):
        ang = i * math.pi / 4.0
        _line(dl, x, y, s, (0.5 + math.cos(ang) * 0.24, 0.5 + math.sin(ang) * 0.24),
              (0.5 + math.cos(ang) * 0.42, 0.5 + math.sin(ang) * 0.42), col, 2.0)
    dl.add_circle(cx, cy, r, col, 24, 2.0)
    dl.add_circle_filled(cx, cy, s * 0.10, col, 16)


def plus(dl, x, y, s, col):
    _line(dl, x, y, s, (0.5, 0.2), (0.5, 0.8), col, 2.4)
    _line(dl, x, y, s, (0.2, 0.5), (0.8, 0.5), col, 2.4)


def sun(dl, x, y, s, col):
    cx, cy = x + s * 0.5, y + s * 0.5
    dl.add_circle_filled(cx, cy, s * 0.17, col, 20)
    for i in range(8):
        ang = i * math.pi / 4.0
        _line(dl, x, y, s, (0.5 + math.cos(ang) * 0.30, 0.5 + math.sin(ang) * 0.30),
              (0.5 + math.cos(ang) * 0.44, 0.5 + math.sin(ang) * 0.44), col, 1.8)


def play(dl, x, y, s, col):
    dl.add_triangle_filled(x + s * 0.30, y + s * 0.22, x + s * 0.30, y + s * 0.78,
                           x + s * 0.80, y + s * 0.50, col)


def power(dl, x, y, s, col):
    cx, cy = x + s * 0.5, y + s * 0.5
    dl.add_circle(cx, cy, s * 0.30, col, 24, 2.0)
    _line(dl, x, y, s, (0.5, 0.12), (0.5, 0.48), col, 2.2)


def close_x(dl, x, y, s, col):
    _line(dl, x, y, s, (0.28, 0.28), (0.72, 0.72), col, 2.2)
    _line(dl, x, y, s, (0.72, 0.28), (0.28, 0.72), col, 2.2)


def trash(dl, x, y, s, col):
    dl.add_rect(x + s * 0.30, y + s * 0.32, x + s * 0.70, y + s * 0.78, col, 0.0, 0, 1.8)
    _line(dl, x, y, s, (0.22, 0.32), (0.78, 0.32), col, 2.0)
    _line(dl, x, y, s, (0.42, 0.24), (0.58, 0.24), col, 2.0)
    _line(dl, x, y, s, (0.42, 0.40), (0.42, 0.70), col, 1.4)
    _line(dl, x, y, s, (0.58, 0.40), (0.58, 0.70), col, 1.4)


def move(dl, x, y, s, col):
    _line(dl, x, y, s, (0.5, 0.15), (0.5, 0.85), col, 1.8)
    _line(dl, x, y, s, (0.15, 0.5), (0.85, 0.5), col, 1.8)
    for a, b, c in (((0.5, 0.12), (0.43, 0.24), (0.57, 0.24)),
                    ((0.5, 0.88), (0.43, 0.76), (0.57, 0.76)),
                    ((0.12, 0.5), (0.24, 0.43), (0.24, 0.57)),
                    ((0.88, 0.5), (0.76, 0.43), (0.76, 0.57))):
        dl.add_triangle_filled(x + a[0] * s, y + a[1] * s, x + b[0] * s, y + b[1] * s,
                               x + c[0] * s, y + c[1] * s, col)


def rotate(dl, x, y, s, col):
    cx, cy = x + s * 0.5, y + s * 0.5
    r = s * 0.30
    dl.path_clear() if hasattr(dl, "path_clear") else None
    # Draw a 3/4 arc as short segments, then an arrowhead.
    pts = []
    for i in range(0, 19):
        a = math.radians(-40 + i * 15)
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    for p, q in zip(pts, pts[1:]):
        dl.add_line(p[0], p[1], q[0], q[1], col, 2.0)
    ex, ey = pts[-1]
    dl.add_triangle_filled(ex, ey - s * 0.09, ex + s * 0.10, ey, ex - s * 0.02, ey + s * 0.06, col)


def scale(dl, x, y, s, col):
    _line(dl, x, y, s, (0.25, 0.75), (0.72, 0.28), col, 1.8)
    dl.add_rect_filled(x + s * 0.16, y + s * 0.66, x + s * 0.30, y + s * 0.80, col)
    dl.add_rect(x + s * 0.66, y + s * 0.20, x + s * 0.82, y + s * 0.36, col, 0.0, 0, 1.8)


def scale_all(dl, x, y, s, col):
    cx, cy = x + s * 0.5, y + s * 0.5
    dl.add_circle(cx, cy, s * 0.12, col, 16, 1.8)
    for ang in (0.25, 0.75, 1.25, 1.75):
        a = ang * math.pi
        _line(dl, x, y, s, (0.5 + math.cos(a) * 0.20, 0.5 + math.sin(a) * 0.20),
              (0.5 + math.cos(a) * 0.42, 0.5 + math.sin(a) * 0.42), col, 1.8)


def cube(dl, x, y, s, col):
    dl.add_rect(x + s * 0.28, y + s * 0.34, x + s * 0.66, y + s * 0.72, col, 0.0, 0, 1.8)
    _line(dl, x, y, s, (0.28, 0.34), (0.40, 0.24), col, 1.6)
    _line(dl, x, y, s, (0.66, 0.34), (0.78, 0.24), col, 1.6)
    _line(dl, x, y, s, (0.66, 0.72), (0.78, 0.62), col, 1.6)
    _line(dl, x, y, s, (0.40, 0.24), (0.78, 0.24), col, 1.6)
    _line(dl, x, y, s, (0.78, 0.24), (0.78, 0.62), col, 1.6)


def sphere(dl, x, y, s, col):
    cx, cy = x + s * 0.5, y + s * 0.5
    dl.add_circle(cx, cy, s * 0.28, col, 28, 1.8)
    # a couple of latitude/longitude arcs to read as a sphere
    dl.add_circle(cx, cy, s * 0.28, col, 28, 1.0)
    for p, q in (((0.22, 0.5), (0.78, 0.5)),):
        _line(dl, x, y, s, p, q, col, 1.0)


def cone(dl, x, y, s, col):
    dl.add_triangle(x + s * 0.5, y + s * 0.22, x + s * 0.26, y + s * 0.74,
                    x + s * 0.74, y + s * 0.74, col, 1.8)
    _line(dl, x, y, s, (0.26, 0.74), (0.74, 0.74), col, 1.6)


def cylinder(dl, x, y, s, col):
    dl.add_line(x + s * 0.30, y + s * 0.28, x + s * 0.30, y + s * 0.72, col, 1.8)
    dl.add_line(x + s * 0.70, y + s * 0.28, x + s * 0.70, y + s * 0.72, col, 1.8)
    cx = x + s * 0.5
    dl.add_circle(cx, y + s * 0.28, s * 0.20, col, 20, 1.6)  # rough ellipse top
    dl.add_circle(cx, y + s * 0.72, s * 0.20, col, 20, 1.6)


def plane(dl, x, y, s, col):
    dl.add_line(x + s * 0.18, y + s * 0.62, x + s * 0.52, y + s * 0.40, col, 1.8)
    dl.add_line(x + s * 0.52, y + s * 0.40, x + s * 0.86, y + s * 0.62, col, 1.8)
    dl.add_line(x + s * 0.86, y + s * 0.62, x + s * 0.52, y + s * 0.84, col, 1.8)
    dl.add_line(x + s * 0.52, y + s * 0.84, x + s * 0.18, y + s * 0.62, col, 1.8)


def bulb(dl, x, y, s, col):
    cx, cy = x + s * 0.5, y + s * 0.44
    dl.add_circle(cx, cy, s * 0.22, col, 22, 1.8)
    dl.add_rect(x + s * 0.42, y + s * 0.66, x + s * 0.58, y + s * 0.78, col, 0.0, 0, 1.6)
    for i in range(6):
        ang = i * math.pi / 3.0
        _line(dl, x, y, s, (0.5 + math.cos(ang) * 0.30, 0.44 + math.sin(ang) * 0.30),
              (0.5 + math.cos(ang) * 0.40, 0.44 + math.sin(ang) * 0.40), col, 1.2)


def spot(dl, x, y, s, col):
    dl.add_circle_filled(x + s * 0.5, y + s * 0.24, s * 0.09, col, 14)
    dl.add_triangle(x + s * 0.42, y + s * 0.30, x + s * 0.22, y + s * 0.80,
                    x + s * 0.78, y + s * 0.80, col, 1.8)


def copy(dl, x, y, s, col):
    dl.add_rect(x + s * 0.36, y + s * 0.24, x + s * 0.74, y + s * 0.62, col, 0.0, 0, 1.6)
    dl.add_rect(x + s * 0.26, y + s * 0.38, x + s * 0.64, y + s * 0.76, col, 0.0, 0, 1.6)


def paste(dl, x, y, s, col):
    dl.add_rect(x + s * 0.28, y + s * 0.26, x + s * 0.72, y + s * 0.80, col, 0.0, 0, 1.6)
    dl.add_rect_filled(x + s * 0.42, y + s * 0.20, x + s * 0.58, y + s * 0.30, col)


def panel(dl, x, y, s, col):
    dl.add_rect(x + s * 0.24, y + s * 0.26, x + s * 0.76, y + s * 0.74, col, 0.0, 0, 1.8)
    _line(dl, x, y, s, (0.24, 0.40), (0.76, 0.40), col, 1.4)
    _line(dl, x, y, s, (0.40, 0.40), (0.40, 0.74), col, 1.4)


def icon_button(id_str: str, draw_fn, size: float = 26.0, tooltip: str | None = None,
                enabled: bool = True, active: bool = False) -> bool:
    """An icon-only toolbar button. Returns True on a click (only when
    enabled). `active` tints the background to show a toggled/current state."""
    imgui.push_id(id_str)
    x, y = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button("btn", size, size)
    hovered = imgui.is_item_hovered()
    dl = imgui.get_window_draw_list()
    if active:
        dl.add_rect_filled(x, y, x + size, y + size, _col(_ACTIVE_BG), 4.0)
    elif hovered and enabled:
        dl.add_rect_filled(x, y, x + size, y + size, _col(_HOVER_BG), 4.0)
    col = _col(_FG if enabled else _FG_DISABLED)
    draw_fn(dl, x, y, size, col)
    if tooltip and hovered:
        imgui.set_tooltip(tooltip)
    imgui.pop_id()
    return clicked and enabled
