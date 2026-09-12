"""Material preset buttons for the Properties panel - thin wrapper around
engine/material_textures.py's apply_material_preset, so the panel doesn't
need to know anything about how a preset is generated or uploaded."""
from __future__ import annotations

import moderngl

from engine.scene import Material
from engine.material_textures import apply_material_preset

# (button label, procedural_textures.py preset key)
PRESETS = (
    ("Beton", "concrete"),
    ("Metal brosse", "brushed_metal"),
    ("Caoutchouc", "rubber"),
    ("Platre", "plaster"),
)


def apply_preset(ctx: moderngl.Context, material: Material, preset_name: str) -> None:
    apply_material_preset(ctx, material, preset_name)
