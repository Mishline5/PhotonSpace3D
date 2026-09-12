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
    // Appended after u_flags, not interleaved earlier: prepass.vert and
    // volumetric.frag declare this same block but stop at u_flags, so their
    // offsets for everything above stay correct without touching those files.
    vec4 u_ambient_sky;        // rgb tint, a = intensity
    vec4 u_ambient_ground;    // rgb tint, a unused
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

// Material textures, not scalars: always sampled, never branched on - a
// material without a real texture gets a 1x1 default encoding its old
// scalar value (see engine/material_textures.py), the same "one code path,
// level/absence reproduces old behavior exactly" discipline already used
// for RT sample count and fog density.
uniform sampler2D u_albedo_map;
uniform sampler2D u_roughness_map;
uniform sampler2D u_metallic_map;
uniform sampler2D u_normal_map;
uniform float u_reflectivity;
uniform sampler2D u_shadow_map;
uniform sampler2D u_ao_map;
uniform int u_shadow_blocker_radius; // quality-level-driven PCSS blocker-search radius, texels (settings.py)
uniform int u_shadow_max_radius;     // quality-level-driven PCSS max penumbra/PCF radius, texels (settings.py)
uniform float u_shadow_near;         // per-frame: near plane of the scene-fitted shadow frustum (shadow_pass.py)
uniform float u_shadow_depth_range;  // per-frame: far - near of that same frustum
uniform int u_rt_samples;   // quality-level-driven reflection sample count (settings.py)
uniform float u_fog_density; // quality-level-driven atmospheric extinction coefficient (settings.py)

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

out vec4 frag_color;

const float PI = 3.14159265359;

// --- shadow mapping -------------------------------------------------------

// Average depth (and count) of shadow-map texels closer to the light than
// `receiver_depth`, searched in a `search_r`-texel window - the PCSS
// "blocker search" step. A zero count means nothing nearby occludes this
// point, so the caller can skip the PCF filter entirely (fully lit).
vec2 find_blocker(vec2 uv, float receiver_depth, int search_r) {
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float sum = 0.0;
    float count = 0.0;
    for (int x = -search_r; x <= search_r; x++) {
        for (int y = -search_r; y <= search_r; y++) {
            float d = texture(u_shadow_map, uv + vec2(x, y) * texel).r;
            if (d < receiver_depth) {
                sum += d;
                count += 1.0;
            }
        }
    }
    return vec2(count > 0.0 ? sum / count : 0.0, count);
}

// PCSS against the directional-light shadow map: penumbra radius grows with
// (receiver-to-blocker distance) / (blocker-to-light distance), the
// standard similarity-triangle estimate - exact here, not approximate,
// because this shadow map is orthographic, so its device depth is already
// linear in light-space world distance (unlike the usual perspective-
// shadow-map case, which needs to linearize depth first). Slope-scaled bias
// fights shadow acne without a fixed offset thick enough to peter-pan
// (detach) the shadow from its caster.
float sample_shadow(vec3 world_pos, float NdotL) {
    vec4 light_clip = u_light_view_proj * vec4(world_pos, 1.0);
    vec3 proj = light_clip.xyz / light_clip.w;
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0) {
        return 1.0; // beyond the shadow frustum's far plane: fully lit
    }
    float bias = max(0.0025 * (1.0 - NdotL), 0.0006);
    float receiver_depth = proj.z - bias;

    vec2 blocker = find_blocker(proj.xy, receiver_depth, u_shadow_blocker_radius);
    if (blocker.y < 1.0) {
        return 1.0; // no blockers nearby: fully lit, skip the PCF filter below entirely
    }

    // u_dir_light_dir.w carries the light's softness dial (see scene.py's
    // DirectionalLight.softness / frame_ubo.py's pad reuse) - a dimensionless
    // "how big is this light" knob, not a physical angular size.
    float depth_diff_world = (receiver_depth - blocker.x) * u_shadow_depth_range;
    float blocker_world_dist = u_shadow_near + blocker.x * u_shadow_depth_range;
    float map_size = float(textureSize(u_shadow_map, 0).x);
    float penumbra_texels = depth_diff_world * u_dir_light_dir.w * map_size
                             / (2.0 * max(blocker_world_dist, 1e-3));
    int r = int(clamp(penumbra_texels, 1.0, float(u_shadow_max_radius)));

    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float lit = 0.0;
    float taps = 0.0;
    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            float closest = texture(u_shadow_map, proj.xy + vec2(x, y) * texel).r;
            lit += (receiver_depth > closest) ? 0.0 : 1.0;
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

// Karis's closed-form fit of the GGX split-sum directional albedo (the same
// curve an environment-BRDF LUT would encode) - used here only to recover
// the energy single-scatter GGX loses at high roughness, not for any IBL
// term (this scene has none). No LUT/bake pass needed, GL 4.1-friendly.
vec2 env_brdf_approx(float NdotV, float roughness) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Kulla-Conty style multi-scatter compensation: single-scatter GGX loses
// energy as roughness grows (Smith's G masks/shadows a lobe fraction that
// physically re-emerges via inter-facet bounces), which makes rough metals
// look darker than they should. `Ess` (single-scatter albedo at F0 = white)
// is the missing-energy signal; boosting by F0 keeps dielectrics (F0~0.04)
// essentially unaffected while rough metals brighten correctly.
vec3 energy_compensation(vec3 F0, float roughness, float NdotV) {
    vec2 ab = env_brdf_approx(NdotV, roughness);
    float Ess = ab.x + ab.y;
    return 1.0 + F0 * (1.0 / max(Ess, 0.05) - 1.0);
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
    specular *= energy_compensation(F0, roughness, NdotV);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    return (diffuse + specular) * radiance * NdotL;
}

// --- ambient (indirect-light stand-in) -------------------------------------

// Two-color hemisphere (sky above, ground-bounce below, blended by N.y)
// replacing the old flat vec3(0.05)*albedo constant - still not a real
// environment capture, but it varies with surface orientation the way real
// indirect light does, at essentially the same cost as the constant it
// replaces. Shared by main() (AO-modulated there) and trace_reflection()
// (not AO-modulated there, same pre-existing limitation as before).
vec3 ambient_hemisphere(vec3 N, vec3 albedo, float metallic, vec3 F0) {
    float sky_weight = N.y * 0.5 + 0.5;
    vec3 irradiance = mix(u_ambient_ground.rgb, u_ambient_sky.rgb, sky_weight) * u_ambient_sky.a;
    vec3 kD = (1.0 - F0) * (1.0 - metallic);
    return kD * albedo * irradiance;
}

// --- normal mapping (screen-space derivatives, no vertex tangent attribute) -

// Reconstructs a per-pixel tangent frame from screen-space derivatives of
// world position and UV (Lengyel's method) instead of a precomputed vertex
// tangent attribute - deliberately, so texturing needed zero changes to
// mesh.py/primitives.py's vertex format (VERTEX_DTYPE stays position+
// normal+uv). Cheaper to add, and well-suited to this project's
// deliberately subtle/low-frequency procedural textures (concrete grain,
// brushed streaks) - exactly the regime where this technique's known
// weakness (instability on high-frequency detail at grazing angles) is
// least likely to show. Forward-compatible with any future mesh that only
// ever carries position/normal/uv.
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv) {
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float inv_max = inversesqrt(max(dot(T, T), dot(B, B)));
    return mat3(T * inv_max, B * inv_max, N);
}

// `tangent_normal` decoded straight from a sampled normal-map texel
// ([0,1] -> [-1,1]). The default 1x1 normal map is (0.5, 0.5, 1.0), which
// decodes to (0, 0, 1) - TBN * (0,0,1) == N exactly, a true no-op for any
// material that hasn't opted into a real normal map.
vec3 apply_normal_map(vec3 N, vec3 world_pos, vec2 uv, vec3 tangent_normal) {
    mat3 TBN = cotangent_frame(N, world_pos, uv);
    return normalize(TBN * tangent_normal);
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

// Finite cylinder about the local Y axis (matches make_cylinder's geometry:
// radius in X/Z, capped flat disks at y = +-half_height). Side is the
// standard quadratic in x/z; each of its two roots is independently
// height-clipped (rather than picking the nearer root first and only then
// checking height) because a ray can radially enter/exit the infinite tube
// entirely above or below the finite cylinder while still crossing a cap in
// between - checking both roots, plus both caps, and keeping the overall
// nearest valid hit handles that case correctly instead of missing it.
float intersect_cylinder_local(vec3 ro, vec3 rd, float radius, float half_height,
                                out vec3 local_hit, out vec3 local_normal) {
    float closest_t = 1e30;
    vec3 hit_pos = vec3(0.0);
    vec3 hit_normal = vec3(0.0, 1.0, 0.0);

    float a = rd.x * rd.x + rd.z * rd.z;
    if (a > 1e-8) {
        float b = 2.0 * (ro.x * rd.x + ro.z * rd.z);
        float c = ro.x * ro.x + ro.z * ro.z - radius * radius;
        float disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
            float sqrt_disc = sqrt(disc);
            float t0 = (-b - sqrt_disc) / (2.0 * a);
            float t1 = (-b + sqrt_disc) / (2.0 * a);
            if (t0 > 1e-4 && t0 < closest_t) {
                vec3 hit = ro + rd * t0;
                if (abs(hit.y) <= half_height) {
                    closest_t = t0;
                    hit_pos = hit;
                    hit_normal = normalize(vec3(hit.x, 0.0, hit.z));
                }
            }
            if (t1 > 1e-4 && t1 < closest_t) {
                vec3 hit = ro + rd * t1;
                if (abs(hit.y) <= half_height) {
                    closest_t = t1;
                    hit_pos = hit;
                    hit_normal = normalize(vec3(hit.x, 0.0, hit.z));
                }
            }
        }
    }

    if (abs(rd.y) > 1e-6) {
        float t_bottom = (-half_height - ro.y) / rd.y;
        if (t_bottom > 1e-4 && t_bottom < closest_t) {
            vec3 hit = ro + rd * t_bottom;
            if (hit.x * hit.x + hit.z * hit.z <= radius * radius) {
                closest_t = t_bottom;
                hit_pos = hit;
                hit_normal = vec3(0.0, -1.0, 0.0);
            }
        }
        float t_top = (half_height - ro.y) / rd.y;
        if (t_top > 1e-4 && t_top < closest_t) {
            vec3 hit = ro + rd * t_top;
            if (hit.x * hit.x + hit.z * hit.z <= radius * radius) {
                closest_t = t_top;
                hit_pos = hit;
                hit_normal = vec3(0.0, 1.0, 0.0);
            }
        }
    }

    if (closest_t >= 1e30) {
        return -1.0;
    }
    local_hit = hit_pos;
    local_normal = hit_normal;
    return closest_t;
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
vec3 trace_reflection(vec3 world_origin, vec3 world_dir, bool apply_shadow) {
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
        } else if (kind == 3) {
            t = intersect_cylinder_local(ro, rd, u_primitives[i].half_extents_pad.x,
                                          u_primitives[i].half_extents_pad.y, lh, ln);
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

    vec3 hit_world_pos = world_origin + world_dir * closest_t;
    vec3 hit_V = -world_dir;
    vec3 hit_L_dir = normalize(-u_dir_light_dir.xyz);
    // Shadow the reflection hit point itself, not just the direct-lit
    // surface that reflects it - retires the "reflections show a fully-lit
    // duplicate of the scene" limitation (an object sitting in shadow now
    // correctly appears shadowed in a reflection too). Measured to be the
    // single biggest GPU-cost jump in this pass once the scene grew to 8 RT
    // primitives over mostly-reflective ground/walls (see benchmark.py) - a
    // full PCSS blocker-search-plus-PCF per jittered reflection sample
    // (up to 4x at rt_level 3) pushed the whole scene below 60 FPS
    // uncapped. `apply_shadow` restricts this to the mirror-direction
    // sample (index 0) only; the jittered soft-reflection samples fall back
    // to unshadowed, same as before this feature existed.
    float hit_shadow = 1.0;
    if (apply_shadow && u_flags.y != 0) {
        hit_shadow = sample_shadow(hit_world_pos, max(dot(hit_normal, hit_L_dir), 0.0));
    }
    vec3 direct = shade_light(hit_normal, hit_V, hit_L_dir,
                               u_dir_light_color.rgb * u_dir_light_color.a * hit_shadow,
                               hit_albedo, hit_metallic, hit_roughness, hit_F0);
    vec3 ambient = ambient_hemisphere(hit_normal, hit_albedo, hit_metallic, hit_F0);
    return direct + ambient;
}

void main() {
    vec3 albedo = texture(u_albedo_map, v_uv).rgb;
    float roughness = clamp(texture(u_roughness_map, v_uv).r, 0.03, 1.0);
    float metallic = texture(u_metallic_map, v_uv).r;

    vec3 geometric_N = normalize(v_normal);
    vec3 tangent_normal = texture(u_normal_map, v_uv).xyz * 2.0 - 1.0;
    vec3 N = apply_normal_map(geometric_N, v_world_pos, v_uv, tangent_normal);
    vec3 V = normalize(u_cam_pos.xyz - v_world_pos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 color = vec3(0.0);

    // Directional light, attenuated by the shadow map (only this light
    // casts shadows - see shadow_pass.py for why).
    vec3 L_dir = normalize(-u_dir_light_dir.xyz);
    float shadow = 1.0;
    if (u_flags.y != 0) {
        shadow = sample_shadow(v_world_pos, max(dot(N, L_dir), 0.0));
    }
    vec3 dir_radiance = u_dir_light_color.rgb * u_dir_light_color.a * shadow;
    color += shade_light(N, V, L_dir, dir_radiance, albedo, metallic, roughness, F0);

    // Point light: windowed inverse-square falloff, reaching exactly zero
    // at its declared range so it has a well-defined radius of effect.
    vec3 to_point = u_point_light_pos.xyz - v_world_pos;
    float dist = length(to_point);
    vec3 L_point = to_point / max(dist, 1e-4);
    float range = max(u_point_light_pos.a, 1e-4);
    float window = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    float falloff = (window * window) / (dist * dist + 1.0);
    vec3 point_radiance = u_point_light_color.rgb * u_point_light_color.a * falloff;
    color += shade_light(N, V, L_point, point_radiance, albedo, metallic, roughness, F0);

    // Hemisphere ambient standing in for indirect/IBL lighting - the only
    // term AO is allowed to modulate (constraint: AO must never darken direct light).
    float ao = 1.0;
    if (u_flags.x != 0) {
        vec2 screen_uv = gl_FragCoord.xy / u_time_info.zw; // zw = screen_w, screen_h
        ao = texture(u_ao_map, screen_uv).r;
    }
    color += ambient_hemisphere(N, albedo, metallic, F0) * ao;

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
            vec3 jittered = jitter_reflection(R, roughness, s, v_world_pos);
            reflection_sum += trace_reflection(v_world_pos + N * 0.01, jittered, s == 0);
        }
        vec3 reflection = reflection_sum / float(u_rt_samples);
        float grazing = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
        float reflect_weight = mix(F0.r, 1.0, grazing) * (1.0 - roughness) * u_reflectivity;
        color += reflection * reflect_weight;
    }

    // Atmospheric extinction (Beer-Lambert on camera-to-fragment distance) -
    // not a particle system, just aerial-perspective fade toward a fog color
    // derived from the ambient hemisphere + a directional sun-glow term
    // (never an arbitrary color - a real fog takes its tint from whatever
    // light is passing through it). At u_fog_density == 0 this mix is the
    // identity, so no separate enabled/disabled flag is needed.
    vec3 view_dir = normalize(v_world_pos - u_cam_pos.xyz);
    float sun_glow = pow(max(dot(view_dir, normalize(-u_dir_light_dir.xyz)), 0.0), 8.0);
    vec3 fog_color = mix(u_ambient_ground.rgb, u_ambient_sky.rgb, 0.5) * u_ambient_sky.a
                    + u_dir_light_color.rgb * u_dir_light_color.a * sun_glow * 0.5;
    float fog_amount = 1.0 - exp(-length(v_world_pos - u_cam_pos.xyz) * u_fog_density);
    color = mix(color, fog_color, fog_amount);

    frag_color = vec4(color, 1.0);
}
