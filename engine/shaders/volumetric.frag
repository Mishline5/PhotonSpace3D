#version 410 core

// Screen-space volumetric light shafts ("god rays") for the directional
// light: raymarches from the camera to each pixel's actual surface depth,
// accumulating in-scattered light only where the shadow map says that point
// along the ray is lit. Additively blended into the HDR color target (see
// volumetric_pass.py) - never darkens anything, only adds light, so it
// composes safely with everything already in the buffer.

in vec2 v_uv;
out vec4 frag_color;

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

uniform sampler2D u_depth_map;   // forward pass's own scene depth (device depth)
uniform sampler2D u_shadow_map;
uniform mat4 u_inv_view_proj;
uniform int u_steps;
uniform float u_density;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float shadow_at(vec3 world_pos) {
    vec4 light_clip = u_light_view_proj * vec4(world_pos, 1.0);
    vec3 proj = light_clip.xyz / light_clip.w;
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0) {
        return 1.0;
    }
    float closest = texture(u_shadow_map, proj.xy).r;
    return proj.z - 0.0015 > closest ? 0.0 : 1.0;
}

void main() {
    float device_depth = texture(u_depth_map, v_uv).r;

    // A pixel where the forward pass drew no geometry keeps the depth
    // target's cleared value (1.0, the far plane) - marching a ray all the
    // way out to the 100-unit far clip with nothing to occlude it saturates
    // the Beer-Lambert term below to ~1 (mostly-unoccluded, "lit" whenever
    // it's outside the light's own shadow frustum) and paints the entire
    // empty sky at near-full light color instead of contributing nothing.
    // No skybox exists in this minimal demo (see forward.frag's
    // trace_reflection escaped-ray comment) - "nothing there" must mean "no
    // volumetric contribution", not "an unbounded lit path".
    if (device_depth >= 0.9999) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec4 near_clip = vec4(v_uv * 2.0 - 1.0, -1.0, 1.0);
    vec4 far_clip = vec4(v_uv * 2.0 - 1.0, device_depth * 2.0 - 1.0, 1.0);
    vec4 near_world4 = u_inv_view_proj * near_clip;
    vec4 far_world4 = u_inv_view_proj * far_clip;
    vec3 near_world = near_world4.xyz / near_world4.w;
    vec3 far_world = far_world4.xyz / far_world4.w;

    // Dither the marching start offset per pixel: trades a fixed step
    // count's banding artifacts for high-frequency noise instead, which
    // reads as much less objectionable at low step counts.
    float jitter = hash12(gl_FragCoord.xy);

    float segment_length = length(far_world - near_world);
    float step_size = segment_length / float(u_steps);
    vec3 step_vec = (far_world - near_world) / float(u_steps);
    vec3 pos = near_world + step_vec * jitter;

    // Accumulate lit path length (world units), not a plain 0..1 average:
    // a ray mostly in shadow with one short lit gap must contribute far
    // less than one almost entirely in light, which a plain average of
    // per-step booleans would not distinguish from a uniformly half-lit ray.
    float lit_path_length = 0.0;
    for (int i = 0; i < u_steps; i++) {
        lit_path_length += shadow_at(pos) * step_size;
        pos += step_vec;
    }

    // Saturating (Beer-Lambert-style) response: bounded in [0, 1) no matter
    // how long the ray is, unlike a plain `lit_path_length * density` term
    // which would grow without limit for any pixel that stares far into an
    // unoccluded, fully-lit direction (e.g. across an open floor) and flood
    // the whole image with light instead of producing a localized shaft.
    float amount = 1.0 - exp(-lit_path_length * u_density);

    vec3 light_color = u_dir_light_color.rgb * u_dir_light_color.a;
    frag_color = vec4(light_color * amount, 1.0);
}
