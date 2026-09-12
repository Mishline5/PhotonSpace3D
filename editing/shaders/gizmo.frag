#version 410 core

// Flat, unlit color - this is tool chrome drawn over the finished scene,
// not scene geometry, so it has no business reading the Frame UBO/lights.

uniform vec3 u_color;
uniform float u_highlight;  // 1.0 normal, >1.0 hovered

out vec4 frag_color;

void main() {
    frag_color = vec4(u_color * u_highlight, 1.0);
}
