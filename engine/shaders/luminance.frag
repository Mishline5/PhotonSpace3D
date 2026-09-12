#version 410 core

// First half of auto-exposure: downsamples the HDR frame into a small
// log2(luminance) capture. Drawn into a target far smaller than the HDR
// source (see settings.AUTOEXPOSURE_PARAMS's capture shift), so each output
// texel already averages a block of input texels by ordinary texture
// minification - the *real* averaging down to a single scene-wide value
// happens next, via the full mip chain built on this texture's result (see
// exposure_pass.py), not in this shader.

in vec2 v_uv;
out float frag_color;

uniform sampler2D u_hdr_color;

void main() {
    vec3 color = texture(u_hdr_color, v_uv).rgb;
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    frag_color = log2(max(luminance, 1e-4));
}
