#version 410 core

// Resolves the linear HDR scene color to the display: exposure -> ACES
// filmic curve -> gamma. Kept as its own pass (rather than baked into
// forward.frag) so every intermediate render target in the pipeline stays
// unambiguously linear HDR, with gamma handled in exactly one auditable place.

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_hdr_color;
uniform float u_exposure;              // manual EV compensation, stacked on top of auto-exposure when enabled
uniform bool u_use_auto_exposure;
uniform sampler2D u_adapted_luminance; // 1x1, exposure_pass.py's ping-ponged output (log2 luminance)
uniform float u_auto_exposure_key;     // target "middle grey" luminance, ~0.18

// Narkowicz ACES fitted curve: same cost as Reinhard, noticeably better
// highlight roll-off, no extra white-point parameter to tune (unlike
// Uncharted2 filmic).
vec3 aces_fitted(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    float exposure = u_exposure;
    if (u_use_auto_exposure) {
        // No parameter value degrades this back to "just the manual slider"
        // the way fog density=0 or an absent material texture does - unlike
        // those, auto-exposure needs an explicit on/off switch.
        float avg_luminance = exp2(texelFetch(u_adapted_luminance, ivec2(0, 0), 0).r);
        exposure *= u_auto_exposure_key / max(avg_luminance, 1e-4);
    }
    vec3 hdr = texture(u_hdr_color, v_uv).rgb * exposure;
    vec3 mapped = aces_fitted(hdr);
    vec3 gamma_corrected = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(gamma_corrected, 1.0);
}
