#version 410 core

// Resolves the linear HDR scene color to the display: exposure -> ACES
// filmic curve -> gamma. Kept as its own pass (rather than baked into
// forward.frag) so every intermediate render target in the pipeline stays
// unambiguously linear HDR, with gamma handled in exactly one auditable place.

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_hdr_color;
uniform float u_exposure;

// Narkowicz ACES fitted curve: same cost as Reinhard, noticeably better
// highlight roll-off, no extra white-point parameter to tune (unlike
// Uncharted2 filmic).
vec3 aces_fitted(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(u_hdr_color, v_uv).rgb * u_exposure;
    vec3 mapped = aces_fitted(hdr);
    vec3 gamma_corrected = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(gamma_corrected, 1.0);
}
