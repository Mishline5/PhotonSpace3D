#version 410 core

// Tiny drifting dust motes. Each particle has a fixed random seed position;
// per frame it drifts slowly (sinusoidal, phase varied by the seed) and is
// wrapped into a box centred on the camera, so the same fixed particle set
// always surrounds the viewer no matter where they move - no per-frame
// re-upload of positions.

uniform mat4 u_view_proj;
uniform vec3 u_cam_pos;
uniform float u_time;
uniform float u_speed;
uniform float u_size;
uniform vec2 u_screen;
uniform float u_box;   // half-extent of the wrap volume around the camera

in vec3 in_seed;       // base position, components in [0, 2*u_box)

void main() {
    vec3 p = in_seed;
    float t = u_time * u_speed;
    // Gentle drift; seed components decorrelate the phases per axis/particle.
    p.x += sin(t * 0.31 + in_seed.z * 6.2831) * 0.35;
    p.y += sin(t * 0.19 + in_seed.x * 6.2831) * 0.22;
    p.z += cos(t * 0.27 + in_seed.y * 6.2831) * 0.35;

    // Wrap into [-u_box, u_box) around the camera.
    vec3 rel = mod(p - u_cam_pos + u_box, 2.0 * u_box) - u_box;
    vec3 world = u_cam_pos + rel;

    vec4 clip = u_view_proj * vec4(world, 1.0);
    gl_Position = clip;
    // Perspective-correct, deliberately small ("toutes petites"): clamp hard.
    gl_PointSize = clamp(u_size * u_screen.y * 0.0016 / max(clip.w, 0.1), 0.8, 4.5);
}
