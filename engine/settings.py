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

# level -> (shadow map resolution, blocker-search radius texels, max penumbra/PCF radius texels)
# PCSS replaces the old fixed-radius PCF wholesale (single code path, no
# "old PCF" kept alongside it) - the search radius is deliberately smaller
# than the max penumbra radius since it only needs to find *whether* nearby
# blockers exist, not resolve the full soft-shadow footprint. Resolutions
# bumped and max penumbra tightened vs. the first pass so shadows read
# deeper and crisper ("pas assez profondes, pas trop dégradées") without
# going razor-hard.
SHADOW_PARAMS = {0: (0, 0, 0), 1: (1024, 2, 2), 2: (2048, 2, 3), 3: (4096, 3, 4)}
# level -> (hemisphere sample count, out of the shader's compiled-in max)
SSAO_PARAMS = {0: (0,), 1: (8,), 2: (16,), 3: (24,)}
# level -> number of jittered reflection rays (1 = sharp mirror-like, more = softer)
RT_PARAMS = {0: (0,), 1: (1,), 2: (2,), 3: (4,)}
# level -> raymarch step count through the shadow map for volumetric scattering
VOLUMETRIC_PARAMS = {0: (0,), 1: (12,), 2: (24,), 3: (48,)}
# level -> atmospheric extinction coefficient (1/world unit). At 0.0 the
# Beer-Lambert blend in forward.frag is algebraically the identity, so this
# needs no separate enabled/disabled flag the way shadows/AO/RT do.
FOG_PARAMS = {0: (0.0,), 1: (0.008,), 2: (0.020,), 3: (0.045,)}
# level -> requested MSAA sample count (further clamped at apply-time against
# this GPU's actual GL_MAX_SAMPLES - see capabilities.py/renderer.py).
MSAA_PARAMS = {0: (0,), 1: (2,), 2: (4,), 3: (8,)}
# level -> HDR-size right-shift for the auto-exposure capture texture (0 =
# off). Higher level -> smaller shift -> bigger capture -> more accurate
# spatial average - the real cost/quality axis here, unlike most of this
# pass's cost which is one fixed 1x1 adapt step regardless of level.
AUTOEXPOSURE_PARAMS = {0: (0,), 1: (5,), 2: (4,), 3: (3,)}
# level -> dust particle count. Tiny drifting motes; 0 = disabled. The size/
# speed/opacity/color of the motes are separate continuous fields below so
# the level is purely "how many".
DUST_PARAMS = {0: (0,), 1: (1500,), 2: (4000,), 3: (9000,)}


@dataclass
class QualitySettings:
    shadow_level: int = 2
    ssao_level: int = 2
    rt_level: int = 2
    volumetric_level: int = 1
    fog_level: int = 1
    msaa_level: int = 2
    auto_exposure_level: int = 2
    dust_level: int = 0                 # 0 = off; dust is opt-in
    # Dust mote tuning (continuous, independent of the count level above).
    dust_size: float = 1.4             # on-screen mote size multiplier (kept small)
    dust_speed: float = 1.0            # drift speed multiplier
    dust_opacity: float = 0.5          # per-mote alpha
    dust_color: tuple[float, float, float] = (0.82, 0.80, 0.74)
    grayscale: bool = False            # black & white post filter
    bodycam: bool = False              # bodycam post filter (game only): vignette + soft edge
    vsync: bool = True
    target_fps: float = 60.0
    exposure: float = 1.0

    def clamp(self) -> None:
        self.shadow_level = max(0, min(MAX_LEVEL, self.shadow_level))
        self.ssao_level = max(0, min(MAX_LEVEL, self.ssao_level))
        self.rt_level = max(0, min(MAX_LEVEL, self.rt_level))
        self.volumetric_level = max(0, min(MAX_LEVEL, self.volumetric_level))
        self.fog_level = max(0, min(MAX_LEVEL, self.fog_level))
        self.msaa_level = max(0, min(MAX_LEVEL, self.msaa_level))
        self.auto_exposure_level = max(0, min(MAX_LEVEL, self.auto_exposure_level))
        self.dust_level = max(0, min(MAX_LEVEL, self.dust_level))

    def dust_params(self) -> dict:
        return {
            "count": DUST_PARAMS[self.dust_level][0],
            "size": self.dust_size,
            "speed": self.dust_speed,
            "opacity": self.dust_opacity,
            "color": tuple(self.dust_color),
        }

    @property
    def shadow_map_size(self) -> int:
        return SHADOW_PARAMS[self.shadow_level][0]

    @property
    def shadow_blocker_search_radius(self) -> int:
        return SHADOW_PARAMS[self.shadow_level][1]

    @property
    def shadow_max_penumbra_radius(self) -> int:
        return SHADOW_PARAMS[self.shadow_level][2]

    @property
    def ssao_samples(self) -> int:
        return SSAO_PARAMS[self.ssao_level][0]

    @property
    def rt_samples(self) -> int:
        return RT_PARAMS[self.rt_level][0]

    @property
    def volumetric_steps(self) -> int:
        return VOLUMETRIC_PARAMS[self.volumetric_level][0]

    @property
    def fog_density(self) -> float:
        return FOG_PARAMS[self.fog_level][0]

    @property
    def autoexposure_capture_shift(self) -> int:
        return AUTOEXPOSURE_PARAMS[self.auto_exposure_level][0]

    @property
    def msaa_samples(self) -> int:
        """Requested sample count, not yet clamped to this GPU's actual
        GL_MAX_SAMPLES - see render_targets.MSAATarget, which does that
        clamp itself against a fresh GL query rather than needing a
        GraphicsCapabilities threaded all the way through the Renderer
        constructor for one value."""
        return MSAA_PARAMS[self.msaa_level][0]


def preset_for_tier(tier: str) -> QualitySettings:
    """Default settings for a freshly detected GPU tier (see capabilities.py).

    The user can still override every value from the in-engine settings
    panel (TAB) - this only picks a sane, adaptive starting point instead of
    guessing or hard-coding one preset for every machine.
    """
    if tier == TIER_HIGH:
        return QualitySettings(shadow_level=3, ssao_level=3, rt_level=3, volumetric_level=2, fog_level=2,
                                msaa_level=3, auto_exposure_level=3)
    if tier == TIER_LOW:
        return QualitySettings(shadow_level=1, ssao_level=1, rt_level=1, volumetric_level=0, fog_level=1,
                                msaa_level=1, auto_exposure_level=1, target_fps=60.0)
    return QualitySettings(shadow_level=2, ssao_level=2, rt_level=2, volumetric_level=1, fog_level=1,
                            msaa_level=2, auto_exposure_level=2)  # medium / unknown


def preset_for_capabilities(caps: GraphicsCapabilities) -> QualitySettings:
    settings = preset_for_tier(caps.tier)
    return settings
