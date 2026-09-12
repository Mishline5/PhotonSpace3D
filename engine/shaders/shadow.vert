#version 410 core

// Depth-only pass from the directional light's point of view - see
// engine/shadow_pass.py. Position only; normal/uv are irrelevant here and
// get skipped as padding by Mesh.vertex_array.

uniform mat4 u_light_view_proj;
uniform mat4 u_model;

in vec3 in_position;

void main() {
    gl_Position = u_light_view_proj * u_model * vec4(in_position, 1.0);
}
