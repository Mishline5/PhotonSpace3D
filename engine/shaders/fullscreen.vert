#version 410 core

// Draws a single oversized triangle covering the whole screen from
// gl_VertexID alone - no vertex buffer needed. Shared by every full-screen
// post-process pass (tonemap, SSAO).

out vec2 v_uv;

void main() {
    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
