#version 410 core

// Hemisphere-kernel SSAO (Crytek/LearnOpenGL-style) from the depth+normal
// prepass. Output is a single raw (noisy) occlusion factor; ssao_blur.frag
// smooths out the per-pixel kernel rotation this relies on to avoid banding.
// This factor is later multiplied ONLY against the ambient/indirect term in
// forward.frag - never against direct light.

in vec2 v_uv;
out float frag_ao;

uniform sampler2D u_normal_map;  // view-space normals (prepass)
uniform sampler2D u_depth_map;   // device depth (prepass)
uniform sampler2D u_noise;       // small tiled texture of random rotation vectors

#define KERNEL_MAX_SIZE 24
uniform vec3 u_kernel[KERNEL_MAX_SIZE];
uniform int u_sample_count;  // quality-level-driven, <= KERNEL_MAX_SIZE (see settings.py)

uniform mat4 u_proj;
uniform mat4 u_inv_proj;
uniform vec2 u_noise_scale;
uniform float u_radius;
uniform float u_bias;
uniform float u_strength;

vec3 reconstruct_view_pos(vec2 uv, float device_depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, device_depth * 2.0 - 1.0, 1.0);
    vec4 view = u_inv_proj * clip;
    return view.xyz / view.w;
}

void main() {
    float device_depth = texture(u_depth_map, v_uv).r;
    if (device_depth >= 1.0) {
        frag_ao = 1.0; // background/sky: nothing to occlude
        return;
    }

    vec3 pos = reconstruct_view_pos(v_uv, device_depth);
    vec3 normal = normalize(texture(u_normal_map, v_uv).xyz);
    vec3 random_vec = normalize(texture(u_noise, v_uv * u_noise_scale).xyz);

    vec3 tangent = normalize(random_vec - normal * dot(random_vec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < u_sample_count; i++) {
        vec3 sample_pos = pos + (TBN * u_kernel[i]) * u_radius;

        vec4 offset = u_proj * vec4(sample_pos, 1.0);
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sample_device_depth = texture(u_depth_map, offset.xy).r;
        vec3 sampled_view_pos = reconstruct_view_pos(offset.xy, sample_device_depth);

        float range_check = smoothstep(0.0, 1.0, u_radius / (abs(pos.z - sampled_view_pos.z) + 1e-4));
        occlusion += (sampled_view_pos.z >= sample_pos.z + u_bias ? 1.0 : 0.0) * range_check;
    }

    float ao = 1.0 - (occlusion / float(max(u_sample_count, 1))) * u_strength;
    frag_ao = clamp(ao, 0.0, 1.0);
}
