#version 410 core

// Cook-Torrance GGX PBR: one directional + one point light, a flat ambient
// term standing in for indirect/IBL lighting (this scene has none), plus an
// experimental hybrid ray-traced reflection term with quality-level-driven
// soft-reflection sampling. "Ray tracing" here means analytic ray-plane/
// ray-box/ray-sphere intersection against the scene's own small list of
// primitives (see rt_primitives.py) - not a BVH over arbitrary meshes.
// That's a deliberate, documented limitation (see design_report.md): GL 4.1
// rules out compute shaders on Apple GPUs (see window.py), and even on a
// GL 4.3+ machine, analytic intersection against a handful of primitives is
// simpler and cheaper than a general mesh-tracing compute pass for this
// engine's scenes.

layout(std140) uniform Frame {
    mat4 u_view;
    mat4 u_proj;
    mat4 u_view_proj;
    mat4 u_light_view_proj;
    vec4 u_cam_pos;
    vec4 u_dir_light_dir;      // xyz: direction light travels (light -> surface)
    vec4 u_dir_light_color;    // rgb: color, a: intensity
    vec4 u_point_light_pos;    // xyz: position, a: range
    vec4 u_point_light_color;  // rgb: color, a: intensity
    vec4 u_time_info;          // time, dt, screen_w, screen_h
    ivec4 u_flags;             // ao_enabled, shadows_enabled, rt_enabled, unused
};

struct Primitive {
    mat4 model;
    vec4 albedo_metallic;          // rgb albedo, a metallic
    vec4 rough_kind_reflect_pad;   // x roughness, y kind (0=plane,1=box), z reflectivity
    vec4 half_extents_pad;         // xyz half-extents (box: full local size; plane: xz bounds)
};

layout(std140) uniform Primitives {
    ivec4 u_primitive_header; // x = count
    Primitive u_primitives[8];
};

uniform vec3 u_albedo;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_reflectivity;
uniform sampler2D u_shadow_map;
uniform sampler2D u_ao_map;
uniform int u_pcf_radius;   // quality-level-driven PCF kernel half-width (settings.py)
uniform int u_rt_samples;   // quality-level-driven reflection sample count (settings.py)

in vec3 v_world_pos;
in vec3 v_normal;

out vec4 frag_color;

const float PI = 3.14159265359;

// --- shadow mapping -------------------------------------------------------

// PCF against the directional-light shadow map, kernel half-width driven by
// quality level (settings.py: 1 -> 3x3, 2 -> 3x3, 3 -> 5x5). Slope-scaled
// bias fights shadow acne without a fixed offset thick enough to peter-pan
// (detach) the shadow from its caster.
float sample_shadow(vec3 world_pos, float NdotL) {
    vec4 light_clip = u_light_view_proj * vec4(world_pos, 1.0);
    vec3 proj = light_clip.xyz / light_clip.w;
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0) {
        return 1.0; // beyond the shadow frustum's far plane: fully lit
    }
    float bias = max(0.0025 * (1.0 - NdotL), 0.0006);
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float lit = 0.0;
    float taps = 0.0;
    int r = u_pcf_radius;
    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            float closest = texture(u_shadow_map, proj.xy + vec2(x, y) * texel).r;
            lit += (proj.z - bias > closest) ? 0.0 : 1.0;
            taps += 1.0;
        }
    }
    return lit / max(taps, 1.0);
}

// --- Cook-Torrance PBR -----------------------------------------------------

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-7);
}

float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometry_smith(float NdotV, float NdotL, float roughness) {
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// One light's contribution; `radiance` already folds in color*intensity*attenuation.
vec3 shade_light(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, float metallic, float roughness, vec3 F0) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        return vec3(0.0);
    }
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 1e-4);

    float D = distribution_ggx(N, H, roughness);
    float G = geometry_smith(NdotV, NdotL, roughness);
    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    return (diffuse + specular) * radiance * NdotL;
}

// --- analytic ray intersection for the hybrid RT pass ---------------------

// Bounded rectangle in the local Y=0 plane (matches the finite ground quad
// mesh - an unbounded plane would reflect past the mesh's visible edges).
float intersect_plane_local(vec3 ro, vec3 rd, vec2 half_extents_xz, out vec3 local_hit) {
    if (abs(rd.y) < 1e-6) {
        return -1.0;
    }
    float t = -ro.y / rd.y;
    if (t <= 1e-4) {
        return -1.0;
    }
    vec3 hit = ro + rd * t;
    if (abs(hit.x) > half_extents_xz.x || abs(hit.z) > half_extents_xz.y) {
        return -1.0;
    }
    local_hit = hit;
    return t;
}

// Standard slab-method ray-AABB test in the box's own local space (so it
// stays correct under the owning object's rotation/scale, applied via
// model/inverse(model) at the call site).
float intersect_box_local(vec3 ro, vec3 rd, vec3 half_extents, out vec3 local_hit, out vec3 local_normal) {
    vec3 inv_rd = 1.0 / rd;
    vec3 t0 = (-half_extents - ro) * inv_rd;
    vec3 t1 = (half_extents - ro) * inv_rd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float t_near = max(max(tmin.x, tmin.y), tmin.z);
    float t_far = min(min(tmax.x, tmax.y), tmax.z);
    if (t_near > t_far || t_far <= 1e-4) {
        return -1.0;
    }
    float t = t_near > 1e-4 ? t_near : t_far;
    local_hit = ro + rd * t;
    vec3 abs_local = abs(local_hit / half_extents);
    if (abs_local.x > abs_local.y && abs_local.x > abs_local.z) {
        local_normal = vec3(sign(local_hit.x), 0.0, 0.0);
    } else if (abs_local.y > abs_local.z) {
        local_normal = vec3(0.0, sign(local_hit.y), 0.0);
    } else {
        local_normal = vec3(0.0, 0.0, sign(local_hit.z));
    }
    return t;
}

// Analytic ray-sphere test (rd assumed normalized): solves
// |ro + t*rd|^2 = radius^2 for the nearest t > 0.
float intersect_sphere_local(vec3 ro, vec3 rd, float radius, out vec3 local_hit, out vec3 local_normal) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    float sqrt_disc = sqrt(disc);
    float t = -b - sqrt_disc;
    if (t <= 1e-4) {
        t = -b + sqrt_disc;
    }
    if (t <= 1e-4) {
        return -1.0;
    }
    local_hit = ro + rd * t;
    local_normal = normalize(local_hit);
    return t;
}

// Cheap per-pixel hash, used only to jitter reflection sample directions
// (not a source of visual noise anywhere else) - no texture lookup needed.
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

// Offsets `dir` within a cone whose angle grows with roughness, so averaging
// several jittered samples approximates a soft (glossy) reflection instead
// of a mirror-sharp one. Sample 0 is always the exact mirror direction, so
// rt_level 1 (one sample) reproduces the original sharp-reflection behavior
// exactly.
vec3 jitter_reflection(vec3 dir, float roughness, int sample_index, vec3 seed) {
    if (sample_index == 0 || roughness < 1e-3) {
        return dir;
    }
    float a = hash13(seed + float(sample_index) * 17.13);
    float b = hash13(seed + float(sample_index) * 91.71 + 7.0);
    float angle = a * 2.0 * PI;
    float radius = b * roughness * 0.35;
    vec3 up = abs(dir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, dir));
    vec3 bitangent = cross(dir, tangent);
    vec3 offset = (tangent * cos(angle) + bitangent * sin(angle)) * radius;
    return normalize(dir + offset);
}

// Traces one reflection ray against the small analytic primitive list and
// shades the hit point with direct lighting only (directional light; no
// shadow/AO/second-bounce lookups) - a deliberate single-bounce
// simplification for a real-time hybrid pass, documented in design_report.md.
vec3 trace_reflection(vec3 world_origin, vec3 world_dir) {
    float closest_t = 1e30;
    int hit_index = -1;
    vec3 hit_local_pos = vec3(0.0);
    vec3 hit_local_normal = vec3(0.0, 1.0, 0.0);

    int count = u_primitive_header.x;
    for (int i = 0; i < count; i++) {
        mat4 model = u_primitives[i].model;
        mat4 inv_model = inverse(model);
        vec3 ro = (inv_model * vec4(world_origin, 1.0)).xyz;
        vec3 rd = normalize((inv_model * vec4(world_dir, 0.0)).xyz);

        int kind = int(u_primitives[i].rough_kind_reflect_pad.y);
        vec3 lh, ln;
        float t;
        if (kind == 0) {
            t = intersect_plane_local(ro, rd, u_primitives[i].half_extents_pad.xz, lh);
            ln = vec3(0.0, 1.0, 0.0);
        } else if (kind == 2) {
            t = intersect_sphere_local(ro, rd, u_primitives[i].half_extents_pad.x, lh, ln);
        } else {
            t = intersect_box_local(ro, rd, u_primitives[i].half_extents_pad.xyz, lh, ln);
        }
        if (t > 0.0 && t < closest_t) {
            closest_t = t;
            hit_index = i;
            hit_local_pos = lh;
            hit_local_normal = ln;
        }
    }

    if (hit_index < 0) {
        return vec3(0.0); // ray escaped the scene: treated as black (no skybox in this minimal demo)
    }

    mat4 hit_model = u_primitives[hit_index].model;
    vec3 hit_normal = normalize(mat3(hit_model) * hit_local_normal);
    vec3 hit_albedo = u_primitives[hit_index].albedo_metallic.rgb;
    float hit_metallic = u_primitives[hit_index].albedo_metallic.a;
    float hit_roughness = u_primitives[hit_index].rough_kind_reflect_pad.x;
    vec3 hit_F0 = mix(vec3(0.04), hit_albedo, hit_metallic);

    vec3 hit_V = -world_dir;
    vec3 hit_L_dir = normalize(-u_dir_light_dir.xyz);
    vec3 direct = shade_light(hit_normal, hit_V, hit_L_dir,
                               u_dir_light_color.rgb * u_dir_light_color.a,
                               hit_albedo, hit_metallic, hit_roughness, hit_F0);
    vec3 ambient = vec3(0.05) * hit_albedo;
    return direct + ambient;
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_cam_pos.xyz - v_world_pos);
    vec3 F0 = mix(vec3(0.04), u_albedo, u_metallic);

    vec3 color = vec3(0.0);

    // Directional light, attenuated by the shadow map (only this light
    // casts shadows - see shadow_pass.py for why).
    vec3 L_dir = normalize(-u_dir_light_dir.xyz);
    float shadow = 1.0;
    if (u_flags.y != 0) {
        shadow = sample_shadow(v_world_pos, max(dot(N, L_dir), 0.0));
    }
    vec3 dir_radiance = u_dir_light_color.rgb * u_dir_light_color.a * shadow;
    color += shade_light(N, V, L_dir, dir_radiance, u_albedo, u_metallic, u_roughness, F0);

    // Point light: windowed inverse-square falloff, reaching exactly zero
    // at its declared range so it has a well-defined radius of effect.
    vec3 to_point = u_point_light_pos.xyz - v_world_pos;
    float dist = length(to_point);
    vec3 L_point = to_point / max(dist, 1e-4);
    float range = max(u_point_light_pos.a, 1e-4);
    float window = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    float falloff = (window * window) / (dist * dist + 1.0);
    vec3 point_radiance = u_point_light_color.rgb * u_point_light_color.a * falloff;
    color += shade_light(N, V, L_point, point_radiance, u_albedo, u_metallic, u_roughness, F0);

    // Flat ambient standing in for indirect/IBL lighting - the only term
    // AO is allowed to modulate (constraint: AO must never darken direct light).
    float ao = 1.0;
    if (u_flags.x != 0) {
        vec2 screen_uv = gl_FragCoord.xy / u_time_info.zw; // zw = screen_w, screen_h
        ao = texture(u_ao_map, screen_uv).r;
    }
    color += vec3(0.05) * u_albedo * ao;

    // Hybrid ray-traced reflection, weighted by Fresnel (grazing angles
    // reflect more, as real materials do) and by (1 - roughness) so rough
    // surfaces don't show mirror-sharp reflections. This weight is what
    // keeps direct specular + reflection from adding up unboundedly on the
    // same pixel (constraint: multiple specular sources must not blow out
    // the image) - the reflection is scaled down rather than added at full
    // strength, and the ACES tonemap pass downstream gracefully compresses
    // whatever HDR headroom is left, instead of hard-clipping.
    if (u_flags.z != 0 && u_reflectivity > 0.0 && u_rt_samples > 0) {
        vec3 R = reflect(-V, N);
        vec3 reflection_sum = vec3(0.0);
        for (int s = 0; s < u_rt_samples; s++) {
            vec3 jittered = jitter_reflection(R, u_roughness, s, v_world_pos);
            reflection_sum += trace_reflection(v_world_pos + N * 0.01, jittered);
        }
        vec3 reflection = reflection_sum / float(u_rt_samples);
        float grazing = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
        float reflect_weight = mix(F0.r, 1.0, grazing) * (1.0 - u_roughness) * u_reflectivity;
        color += reflection * reflect_weight;
    }

    frag_color = vec4(color, 1.0);
}
