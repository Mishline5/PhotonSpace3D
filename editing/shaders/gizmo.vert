#version 410 core

uniform mat4 u_mvp;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
