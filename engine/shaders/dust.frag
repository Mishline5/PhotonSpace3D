#version 410 core

// Soft round mote: a radial falloff over the point sprite so each speck is a
// gentle blob rather than a hard square. Alpha-blended over the HDR scene.

uniform vec3 u_color;
uniform float u_opacity;

out vec4 frag_color;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) {
        discard;
    }
    float a = (1.0 - r2);
    a *= a;  // softer edge
    frag_color = vec4(u_color, u_opacity * a);
}
