#version 410 core

// Resolves the linear HDR scene color to the display: exposure -> ACES
// filmic curve -> gamma, plus optional post filters (grayscale, bodycam).
// Kept as its own pass (rather than baked into forward.frag) so every
// intermediate render target in the pipeline stays unambiguously linear HDR,
// with gamma handled in exactly one auditable place.

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_hdr_color;
uniform float u_exposure;              // manual EV compensation, stacked on top of auto-exposure when enabled
uniform bool u_use_auto_exposure;
uniform sampler2D u_adapted_luminance; // 1x1, exposure_pass.py's ping-ponged output (log2 luminance)
uniform float u_auto_exposure_key;     // target "middle grey" luminance, ~0.18
uniform bool u_grayscale;              // black & white filter
uniform bool u_bodycam;                // bodycam filter (vignette + soft round edge + slight chroma)

// Narkowicz ACES fitted curve: same cost as Reinhard, noticeably better
// highlight roll-off, no extra white-point parameter to tune (unlike
// Uncharted2 filmic).
vec3 aces_fitted(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 sample_scene(vec2 uv, float exposure) {
    vec3 hdr = texture(u_hdr_color, uv).rgb * exposure;
    return aces_fitted(hdr);
}

void main() {
    float exposure = u_exposure;
    if (u_use_auto_exposure) {
        // Adapt toward middle grey, but CLAMP the multiplier: an unclamped
        // key/avg blows the whole frame to white when the average is
        // dominated by a dark region (e.g. staring into a bright light over a
        // dark background). The clamp keeps auto-exposure a gentle nudge, not
        // a runaway - the fix for "everything goes white when I look at a
        // light". Also floors avg so it never divides by ~0.
        float avg_luminance = max(exp2(texelFetch(u_adapted_luminance, ivec2(0, 0), 0).r), 0.02);
        float adapt = clamp(u_auto_exposure_key / avg_luminance, 0.6, 1.8);
        exposure *= adapt;
    }

    vec2 uv = v_uv;
    float chroma = 0.0;
    if (u_bodycam) {
        // Slight barrel-ish pull toward centre + a touch of edge chroma.
        vec2 c = uv - 0.5;
        float r2 = dot(c, c);
        uv = 0.5 + c * (1.0 + 0.06 * r2);
        chroma = 0.004 * r2;
    }

    vec3 mapped;
    if (chroma > 0.0) {
        // Cheap chromatic aberration: shift R/B sample points radially.
        vec2 dir = (uv - 0.5);
        mapped.r = sample_scene(uv + dir * chroma, exposure).r;
        mapped.g = sample_scene(uv, exposure).g;
        mapped.b = sample_scene(uv - dir * chroma, exposure).b;
    } else {
        mapped = sample_scene(uv, exposure);
    }

    if (u_grayscale) {
        float luma = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
        mapped = vec3(luma);
    }

    if (u_bodycam) {
        // Strong vignette darkening toward the edges + a soft round framing,
        // the recognisable "bodycam" look.
        vec2 c = v_uv - 0.5;
        float r = length(c) * 1.4;
        float vignette = smoothstep(1.05, 0.35, r);
        mapped *= mix(0.15, 1.0, vignette);
    }

    vec3 gamma_corrected = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(gamma_corrected, 1.0);
}
