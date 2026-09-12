"""Free-fly camera: WASD + mouse-look, movement scaled by delta-time.

Uses yaw/pitch rather than a general quaternion deliberately - a free-look
camera never rolls, so Euler angles can't gimbal-lock here (pitch is clamped
short of +-90) and are simpler to reason about and debug than a quaternion
for this one case.
"""
from __future__ import annotations

import glm


class Camera:
    def __init__(self, position=(0.0, 1.6, 6.0), yaw: float = -90.0, pitch: float = -10.0,
                 fov_degrees: float = 60.0, near: float = 0.05, far: float = 100.0) -> None:
        self.position = glm.vec3(*position)
        self.yaw = yaw
        self.pitch = pitch
        self.fov_degrees = fov_degrees
        self.near = near
        self.far = far
        self.move_speed = 4.0
        self.sprint_multiplier = 3.0
        self.look_sensitivity = 0.12
        self.aspect = 16.0 / 9.0
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self._update_vectors()

    def _update_vectors(self) -> None:
        yaw_r = glm.radians(self.yaw)
        pitch_r = glm.radians(self.pitch)
        self.front = glm.normalize(glm.vec3(
            glm.cos(yaw_r) * glm.cos(pitch_r),
            glm.sin(pitch_r),
            glm.sin(yaw_r) * glm.cos(pitch_r),
        ))
        world_up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.normalize(glm.cross(self.front, world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def look(self, dx: float, dy: float) -> None:
        self.yaw += dx * self.look_sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch - dy * self.look_sensitivity))
        self._update_vectors()

    def move(self, dt: float, forward: float, right: float, up: float, sprint: bool = False) -> None:
        speed = self.move_speed * (self.sprint_multiplier if sprint else 1.0) * dt
        if forward or right or up:
            direction = self.front * forward + self.right * right + glm.vec3(0.0, 1.0, 0.0) * up
            length = glm.length(direction)
            if length > 1e-6:
                self.position += (direction / length) * speed

    def view_matrix(self) -> glm.mat4:
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def projection_matrix(self) -> glm.mat4:
        return glm.perspective(glm.radians(self.fov_degrees), self.aspect, self.near, self.far)

    def set_aspect(self, width: int, height: int) -> None:
        self.aspect = width / max(height, 1)
