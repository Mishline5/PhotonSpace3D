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
    // Unused in this stage, but must match forward.frag's Frame block
    // field-for-field: the two stages link into one program, and GLSL
    // requires a uniform block referenced by multiple stages to agree
    // exactly, even on members this stage never reads.
    vec4 u_ambient_sky;
    vec4 u_ambient_ground;
};

uniform mat4 u_model;
uniform mat3 u_normal_matrix;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(u_normal_matrix * in_normal);
    v_uv = in_uv;
    gl_Position = u_view_proj * world_pos;
}
