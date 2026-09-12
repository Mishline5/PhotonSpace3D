#version 410 core

// Depth + view-space normal prepass, feeding SSAO before the forward
// lighting pass runs - see engine/prepass.py.

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

out vec3 v_view_normal;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    vec3 world_normal = normalize(u_normal_matrix * in_normal);
    v_view_normal = mat3(u_view) * world_normal;
    gl_Position = u_view_proj * world_pos;
}
