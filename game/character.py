"""Default FPS character: mouse-look camera + ZQSD/WASD ground movement,
gravity, jump, and sphere-vs-mesh collision (game/collision.py). No fly - the
editor's free-fly camera is deliberately disabled in game mode.

The body is a single collision sphere near the feet; the camera rides an eye
offset above its centre. Colliding a sphere (not a box) against real
triangles is what lets a rotated cone or a tilted surface act as a walkable
ramp: the sphere slides up any surface less than vertical instead of being
stopped flat.
"""
from __future__ import annotations

import glm

from engine.camera import Camera
from .collision import resolve_sphere

GRAVITY = 20.0
MOVE_SPEED = 5.5
JUMP_SPEED = 7.0


class FpsCharacter:
    def __init__(self, spawn: tuple[float, float, float], radius: float = 0.4, eye_offset: float = 1.15) -> None:
        self.radius = radius
        self.eye_offset = eye_offset
        self.center = glm.vec3(*spawn)
        self.velocity = glm.vec3(0.0, 0.0, 0.0)
        self.grounded = False

    def update(self, dt: float, camera: Camera, forward: float, strafe: float,
               jump: bool, tris) -> None:
        dt = min(dt, 1.0 / 30.0)  # clamp so a hitch can't tunnel the sphere through geometry

        # Horizontal wish direction from the camera yaw (front/right flattened).
        front = glm.vec3(camera.front.x, 0.0, camera.front.z)
        right = glm.vec3(camera.right.x, 0.0, camera.right.z)
        if glm.length(front) > 1e-4:
            front = glm.normalize(front)
        if glm.length(right) > 1e-4:
            right = glm.normalize(right)
        wish = front * forward + right * strafe
        if glm.length(wish) > 1e-4:
            wish = glm.normalize(wish)

        self.velocity.x = wish.x * MOVE_SPEED
        self.velocity.z = wish.z * MOVE_SPEED
        self.velocity.y -= GRAVITY * dt
        if self.grounded and jump:
            self.velocity.y = JUMP_SPEED

        self.center = self.center + self.velocity * dt
        self.center, grounded, ceiling = resolve_sphere(self.center, self.radius, tris)
        self.grounded = grounded
        if grounded and self.velocity.y < 0.0:
            self.velocity.y = 0.0
        if ceiling and self.velocity.y > 0.0:
            self.velocity.y = 0.0

        if self.center.y < -40.0:  # fell out of the world: respawn above origin
            self.center = glm.vec3(0.0, 6.0, 0.0)
            self.velocity = glm.vec3(0.0, 0.0, 0.0)

        camera.position = glm.vec3(self.center.x, self.center.y + self.eye_offset, self.center.z)
