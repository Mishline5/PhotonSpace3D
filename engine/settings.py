"""Central quality settings: one object the whole renderer reads from.

Shadows, SSAO, hybrid RT and volumetric lighting are each a single level in
[0, 3] - 0 means off (no separate on/off flag needed), 1-3 trade quality for
cost. This module is the only place that maps a level to concrete
parameters (resolution, sample counts, ...); everything else in the engine
just reads `settings.shadow_level` etc. and asks this module what that
level means.
"""
from __future__ import annotations

from dataclasses import dataclass

from .capabilities import GraphicsCapabilities, TIER_LOW, TIER_MEDIUM, TIER_HIGH

MAX_LEVEL = 3

# level -> (shadow map resolution, PCF kernel half-width in texels)
SHADOW_PARAMS = {0: (0, 0), 1: (512, 1), 2: (1024, 1), 3: (2048, 2)}
# level -> (hemisphere sample count, out of the shader's compiled-in max)
SSAO_PARAMS = {0: (0,), 1: (8,), 2: (16,), 3: (24,)}
# level -> number of jittered reflection rays (1 = sharp mirror-like, more = softer)
RT_PARAMS = {0: (0,), 1: (1,), 2: (2,), 3: (4,)}
# level -> raymarch step count through the shadow map for volumetric scattering
VOLUMETRIC_PARAMS = {0: (0,), 1: (12,), 2: (24,), 3: (48,)}


@dataclass
class QualitySettings:
    shadow_level: int = 2
    ssao_level: int = 2
    rt_level: int = 2
    volumetric_level: int = 1
    vsync: bool = True
    target_fps: float = 60.0
    exposure: float = 1.0

    def clamp(self) -> None:
        self.shadow_level = max(0, min(MAX_LEVEL, self.shadow_level))
        self.ssao_level = max(0, min(MAX_LEVEL, self.ssao_level))
        self.rt_level = max(0, min(MAX_LEVEL, self.rt_level))
        self.volumetric_level = max(0, min(MAX_LEVEL, self.volumetric_level))

    @property
    def shadow_map_size(self) -> int:
        return SHADOW_PARAMS[self.shadow_level][0]

    @property
    def shadow_pcf_radius(self) -> int:
        return SHADOW_PARAMS[self.shadow_level][1]

    @property
    def ssao_samples(self) -> int:
        return SSAO_PARAMS[self.ssao_level][0]

    @property
    def rt_samples(self) -> int:
        return RT_PARAMS[self.rt_level][0]

    @property
    def volumetric_steps(self) -> int:
        return VOLUMETRIC_PARAMS[self.volumetric_level][0]


def preset_for_tier(tier: str) -> QualitySettings:
    """Default settings for a freshly detected GPU tier (see capabilities.py).

    The user can still override every value from the in-engine settings
    panel (TAB) - this only picks a sane, adaptive starting point instead of
    guessing or hard-coding one preset for every machine.
    """
    if tier == TIER_HIGH:
        return QualitySettings(shadow_level=3, ssao_level=3, rt_level=3, volumetric_level=2)
    if tier == TIER_LOW:
        return QualitySettings(shadow_level=1, ssao_level=1, rt_level=1, volumetric_level=0, target_fps=60.0)
    return QualitySettings(shadow_level=2, ssao_level=2, rt_level=2, volumetric_level=1)  # medium / unknown


def preset_for_capabilities(caps: GraphicsCapabilities) -> QualitySettings:
    settings = preset_for_tier(caps.tier)
    return settings
