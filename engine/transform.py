"""Object transform with a change-version counter for zero-waste GPU uploads.

`version` increments on every mutation and never resets; the renderer
compares it against the version it last uploaded for that object (see
SceneObject.uploaded_version in scene.py) and only re-writes that object's
tiny per-object uniform when the numbers differ. This is deliberately kept
separate from the internal matrix cache below (_matrix_dirty), which several
independent readers (forward pass, shadow pass, RT primitive extraction) may
consult in the same frame - a plain "dirty flag cleared on read" would only
tell the truth to whichever system happened to read it first.
"""
from __future__ import annotations

import glm


class Transform:
    def __init__(self, position=(0.0, 0.0, 0.0), euler_degrees=(0.0, 0.0, 0.0),
                 scale=(1.0, 1.0, 1.0)) -> None:
        self._position = glm.vec3(*position)
        self._rotation = glm.quat(glm.radians(glm.vec3(*euler_degrees)))
        self._scale = glm.vec3(*scale)
        self._matrix = glm.mat4(1.0)
        self._matrix_dirty = True
        self.version = 0

    def _mark_changed(self) -> None:
        self._matrix_dirty = True
        self.version += 1

    @property
    def position(self) -> glm.vec3:
        return self._position

    @position.setter
    def position(self, value) -> None:
        self._position = glm.vec3(*value)
        self._mark_changed()

    @property
    def scale(self) -> glm.vec3:
        return self._scale

    @scale.setter
    def scale(self, value) -> None:
        self._scale = glm.vec3(*value) if not isinstance(value, (int, float)) else glm.vec3(value)
        self._mark_changed()

    @property
    def rotation(self) -> glm.quat:
        return self._rotation

    @rotation.setter
    def rotation(self, value: glm.quat) -> None:
        self._rotation = value
        self._mark_changed()

    def set_euler_degrees(self, x: float, y: float, z: float) -> None:
        self.rotation = glm.quat(glm.radians(glm.vec3(x, y, z)))

    def _update_matrix(self) -> None:
        t = glm.translate(glm.mat4(1.0), self._position)
        r = glm.mat4_cast(self._rotation)
        s = glm.scale(glm.mat4(1.0), self._scale)
        self._matrix = t * r * s

    def matrix(self) -> glm.mat4:
        if self._matrix_dirty:
            self._update_matrix()
            self._matrix_dirty = False
        return self._matrix
