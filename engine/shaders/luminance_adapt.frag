#version 410 core

// Second half of auto-exposure: a 1x1 draw that reads the coarsest mip of
// luminance.frag's capture (textureLod's `lod` argument is clamped by the
// GL spec to the texture's actual last mip level, so a deliberately huge
// value here always lands on "the single texel averaging the whole
// capture" without this code needing to track mip counts itself), then
// exponentially blends it with the *previous frame's* adapted value -
// frame-rate-independent smoothing (same shape as clock.py's dt-based
// motion), so exposure eases toward a new average instead of snapping to
// it every frame.
//
// Ping-ponged between two 1x1 textures (see exposure_pass.py): this must
// never read and write the same texture in one draw, the same feedback-loop
// hazard volumetric_pass.py's docstring already flags for a different pair.

in vec2 v_uv;
out float frag_color;

uniform sampler2D u_luminance_map;  // full mip chain from luminance.frag
uniform sampler2D u_prev_adapted;   // 1x1, previous frame's blended result
uniform float u_dt;
uniform float u_tau;                // adaptation time constant, seconds

void main() {
    float raw_log = textureLod(u_luminance_map, vec2(0.5), 20.0).r;
    float prev_log = texture(u_prev_adapted, vec2(0.5)).r;
    float alpha = 1.0 - exp(-u_dt / max(u_tau, 1e-4));
    frag_color = mix(prev_log, raw_log, alpha);
}
