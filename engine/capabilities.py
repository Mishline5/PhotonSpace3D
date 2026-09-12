"""Hardware capability detection.

Classifies the GPU so the engine can pick sensible default quality settings
without the user having to tune anything by hand (settings.py turns a tier
into concrete shadow/SSAO/RT/volumetric levels). There is no portable "give
me a performance tier" API in OpenGL itself, so this is a heuristic based on
the GL_VENDOR/GL_RENDERER strings every driver exposes - vendor and GPU
family name substrings are the same signal Windows/Linux/macOS all agree on.

This runs once, right after context creation, and never queries anything
platform-specific beyond standard GL - it works unmodified on an NVIDIA
Windows box, an AMD Linux box, or Apple Silicon.
"""
from __future__ import annotations

from dataclasses import dataclass

import moderngl

TIER_LOW = "low"
TIER_MEDIUM = "medium"
TIER_HIGH = "high"


@dataclass
class GraphicsCapabilities:
    vendor: str
    renderer: str
    gl_version: str
    tier: str  # TIER_LOW | TIER_MEDIUM | TIER_HIGH
    max_texture_size: int
    is_apple_silicon: bool  # informational: this engine's GL 4.1 ceiling applies here specifically


def detect(ctx: moderngl.Context) -> GraphicsCapabilities:
    vendor = str(ctx.info.get("GL_VENDOR") or "")
    renderer = str(ctx.info.get("GL_RENDERER") or "")
    version = str(ctx.info.get("GL_VERSION") or "")
    max_tex = int(ctx.info.get("GL_MAX_TEXTURE_SIZE") or 4096)

    is_apple = "apple" in vendor.lower() or "apple" in renderer.lower()
    tier = _classify(vendor, renderer)
    return GraphicsCapabilities(vendor, renderer, version, tier, max_tex, is_apple)


def _classify(vendor: str, renderer: str) -> str:
    v, r = vendor.lower(), renderer.lower()

    if "nvidia" in v or "geforce" in r or "nvidia" in r or "quadro" in r:
        # RTX/Titan (Turing and newer, real-time-RT-capable hardware) -> high.
        # Older GTX-class -> medium. Anything else NVIDIA -> medium, a safe
        # middle ground rather than assuming the best or the worst.
        if any(tok in r for tok in ("rtx", "titan")):
            return TIER_HIGH
        return TIER_MEDIUM

    if "amd" in v or "amd" in r or "radeon" in r or "ati" in v:
        # RDNA2+ (RX 6000/7000/9000, current-gen consumer/workstation) -> high.
        if any(tok in r for tok in ("rx 6", "rx 7", "rx 9", "radeon pro w", "radeon vii")):
            return TIER_HIGH
        return TIER_MEDIUM

    if "apple" in v or "apple" in r:
        # Good perf/watt and plenty of unified memory bandwidth, but this
        # engine's GL 4.1 ceiling on macOS (see window.py) means the hybrid
        # RT pass never gets a compute-shader path here regardless of chip
        # generation - "medium" reflects the achievable feature set on this
        # platform, not the chip's raw silicon capability.
        return TIER_MEDIUM

    if "intel" in v or "intel" in r:
        # Integrated Intel graphics (UHD/Iris/Arc integrated parts) -> low.
        # Discrete Arc cards would also match this substring; a safe default
        # is still reasonable since we can't distinguish them from the
        # renderer string alone without a much larger lookup table.
        return TIER_LOW

    return TIER_MEDIUM  # unknown hardware: a moderate, safe default


def describe(caps: GraphicsCapabilities) -> str:
    return f"{caps.renderer} ({caps.vendor}) - GL {caps.gl_version} - tier: {caps.tier}"
