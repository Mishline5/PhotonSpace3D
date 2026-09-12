"""Minimal selection state: at most one selected object at a time (matches
"basic, not advanced" scope - no multi-select)."""
from __future__ import annotations

from engine.scene import SceneObject


class Selection:
    def __init__(self) -> None:
        self.object: SceneObject | None = None

    def set(self, obj: SceneObject | None) -> None:
        self.object = obj

    def clear(self) -> None:
        self.object = None

    @property
    def has_selection(self) -> bool:
        return self.object is not None
