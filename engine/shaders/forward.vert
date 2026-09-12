#version 410 core

layout(std140) uniform Frame {
    mat4 u_view;
    mat4 u_proj;
    mat4 u_view_proj;
    mat4 u_light_view_proj;
    vec4 u_cam_pos;
    vec4 u_dir_light_dir;
    vec4 u_dir_light_color;
    vec4 u_point_light_pos;
    vec4 u_point_light_color;
    vec4 u_time_info;
    ivec4 u_flags;
};

uniform mat4 u_model;
uniform mat3 u_normal_matrix;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_world_pos;
out vec3 v_normal;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(u_normal_matrix * in_normal);
    gl_Position = u_view_proj * world_pos;
}
