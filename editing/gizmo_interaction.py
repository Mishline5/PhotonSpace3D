"""Gizmo hover detection (2D, screen-space) and axis-constrained drag math
(3D, world-space) for translate/rotate/scale."""
from __future__ import annotations

import math

import glm

from engine.camera import Camera
from engine.scene import SceneObject

from .gizmo import Gizmo, gizmo_world_size
from .gizmo_geometry import AXIS_WORLD_DIR
from .picking import build_pick_ray


def project_to_screen(point_world: glm.vec3, view_proj: glm.mat4,
                       viewport_size: tuple[int, int]) -> tuple[float, float] | None:
    clip = view_proj * glm.vec4(point_world, 1.0)
    if clip.w <= 1e-5:
        return None  # behind the camera
    ndc = glm.vec3(clip) / clip.w
    w, h = viewport_size
    return ((ndc.x * 0.5 + 0.5) * w, (1.0 - (ndc.y * 0.5 + 0.5)) * h)


def _dist_point_to_segment_2d(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (p[0] - a[0], p[1] - a[1])
    ab_len2 = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if ab_len2 < 1e-9 else max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1]) / ab_len2))
    proj = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return math.hypot(p[0] - proj[0], p[1] - proj[1])


def hover_axis(gizmo: Gizmo, camera: Camera, obj: SceneObject, mode: str,
               mouse_pos: tuple[float, float], viewport_size: tuple[int, int],
               threshold_px: float = 12.0) -> int | None:
    """Nearest handle under `threshold_px`, by distance from the mouse to
    the handle's screen-projected polyline - not a 3D raycast against the
    gizmo's actual triangle geometry (unnecessary precision for a UI
    affordance whose on-screen size is already tightly controlled by
    gizmo_world_size)."""
    view_proj = camera.projection_matrix() * camera.view_matrix()
    apparent_size = gizmo_world_size(camera, glm.vec3(obj.transform.position), viewport_size[1])
    best_axis, best_dist = None, threshold_px

    for axis in (0, 1, 2):
        handle_model = gizmo.handle_matrix(obj, axis, mode, apparent_size)
        if mode == "rotate":
            angles = [i * 2.0 * math.pi / 24 for i in range(25)]
            samples = [glm.vec3(handle_model * glm.vec4(math.cos(t), 0.0, math.sin(t), 1.0)) for t in angles]
        else:
            samples = [
                glm.vec3(handle_model * glm.vec4(0.0, 0.0, 0.0, 1.0)),
                glm.vec3(handle_model * glm.vec4(0.0, 1.0, 0.0, 1.0)),
            ]

        projected = [p for p in (project_to_screen(s, view_proj, viewport_size) for s in samples) if p is not None]
        for a, b in zip(projected, projected[1:]):
            dist = _dist_point_to_segment_2d(mouse_pos, a, b)
            if dist < best_dist:
                best_dist = dist
                best_axis = axis
    return best_axis


def ray_plane_intersect(ro: glm.vec3, rd: glm.vec3, plane_point: glm.vec3, plane_normal: glm.vec3) -> glm.vec3 | None:
    denom = glm.dot(rd, plane_normal)
    if abs(denom) < 1e-6:
        return None
    t = glm.dot(plane_point - ro, plane_normal) / denom
    if t < 0.0:
        return None
    return ro + rd * t


def axis_facing_plane_normal(axis_dir: glm.vec3, camera_pos: glm.vec3, pivot: glm.vec3) -> glm.vec3:
    """Normal of the plane that contains `axis_dir` and faces the camera -
    intersecting the mouse ray with this (fixed for the drag's duration)
    plane is what turns 2D mouse movement into a 1D delta along the axis."""
    to_cam = camera_pos - pivot
    side = glm.cross(axis_dir, to_cam)
    if glm.length(side) < 1e-6:
        # Camera looking (near-)exactly down the axis: to_cam gives no
        # usable "side" direction, so fall back to an arbitrary one not
        # parallel to axis_dir - any plane containing the axis is equally
        # degenerate for dragging in this configuration regardless.
        side = glm.cross(axis_dir, glm.vec3(0.0, 1.0, 0.0))
        if glm.length(side) < 1e-6:
            side = glm.cross(axis_dir, glm.vec3(1.0, 0.0, 0.0))
    return glm.normalize(glm.cross(side, axis_dir))


class DragState:
    """Captured once at drag-start (mouse-down on a handle) and reused every
    frame until mouse-up - the constraint plane is deliberately frozen for
    the whole drag rather than recomputed each frame, since a plane that
    kept following the mouse would no longer constrain anything."""

    def __init__(self, mode: str, axis: int, obj: SceneObject, camera: Camera,
                 mouse_pos: tuple[float, float], viewport_size: tuple[int, int]) -> None:
        self.mode = mode
        self.axis = axis
        self.obj = obj
        self.start_position = glm.vec3(obj.transform.position)
        self.start_rotation = glm.quat(obj.transform.rotation)
        self.start_scale = glm.vec3(obj.transform.scale)
        self.pivot = glm.vec3(obj.transform.position)

        if mode in ("scale", "uniform"):
            self.axis_dir = glm.normalize(obj.transform.rotation * AXIS_WORLD_DIR[axis])
        else:
            self.axis_dir = AXIS_WORLD_DIR[axis]

        ray_origin, ray_dir = _pick_ray(camera, mouse_pos, viewport_size)
        if mode == "rotate":
            self.plane_normal = self.axis_dir
        else:
            self.plane_normal = axis_facing_plane_normal(self.axis_dir, glm.vec3(camera.position), self.pivot)
        self.start_hit = ray_plane_intersect(ray_origin, ray_dir, self.pivot, self.plane_normal)
        self.apparent_size_at_start = gizmo_world_size(camera, self.pivot, viewport_size[1])

    def update(self, camera: Camera, mouse_pos: tuple[float, float], viewport_size: tuple[int, int]) -> None:
        if self.start_hit is None:
            return
        ray_origin, ray_dir = _pick_ray(camera, mouse_pos, viewport_size)
        current_hit = ray_plane_intersect(ray_origin, ray_dir, self.pivot, self.plane_normal)
        if current_hit is None:
            return

        if self.mode == "translate":
            delta = glm.dot(current_hit - self.start_hit, self.axis_dir)
            offset = self.axis_dir * delta
            self.obj.transform.position = tuple(glm.vec3(self.start_position) + offset)

        elif self.mode == "rotate":
            start_dir = glm.normalize(self.start_hit - self.pivot)
            current_dir = glm.normalize(current_hit - self.pivot)
            cos_a = max(-1.0, min(1.0, glm.dot(start_dir, current_dir)))
            angle = math.acos(cos_a)
            sign = glm.dot(glm.cross(start_dir, current_dir), self.axis_dir)
            signed_angle = angle if sign >= 0.0 else -angle
            delta_quat = glm.angleAxis(signed_angle, self.axis_dir)
            # World-space composition (pre-multiply): the ring is drawn in
            # world orientation, so the drag must rotate in world axes too -
            # post-multiplying would rotate in the object's own local axes
            # instead, inconsistent with what's actually on screen.
            self.obj.transform.rotation = delta_quat * self.start_rotation

        elif self.mode == "scale":
            delta = glm.dot(current_hit - self.start_hit, self.axis_dir)
            scale_delta = delta / max(self.apparent_size_at_start, 1e-4)
            new_scale = list(self.start_scale)
            new_scale[self.axis] = max(0.01, self.start_scale[self.axis] * (1.0 + scale_delta))
            self.obj.transform.scale = tuple(new_scale)

        elif self.mode == "uniform":
            # Global scale: one handle's drag scales all three axes equally.
            delta = glm.dot(current_hit - self.start_hit, self.axis_dir)
            factor = max(0.05, 1.0 + delta / max(self.apparent_size_at_start, 1e-4))
            self.obj.transform.scale = (
                max(0.01, self.start_scale.x * factor),
                max(0.01, self.start_scale.y * factor),
                max(0.01, self.start_scale.z * factor),
            )


def _pick_ray(camera: Camera, mouse_pos: tuple[float, float],
              viewport_size: tuple[int, int]) -> tuple[glm.vec3, glm.vec3]:
    return build_pick_ray(camera, mouse_pos, viewport_size)
