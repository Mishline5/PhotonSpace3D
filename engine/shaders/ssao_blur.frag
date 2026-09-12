#version 410 core

// 4x4 box blur matching the SSAO noise tile size - this is what turns the
// per-pixel random kernel rotation from visible dither into smooth-looking
// occlusion.

in vec2 v_uv;
out float frag_ao;

uniform sampler2D u_ao;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_ao, 0));
    float result = 0.0;
    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            result += texture(u_ao, v_uv + vec2(x, y) * texel).r;
        }
    }
    frag_ao = result / 16.0;
}
