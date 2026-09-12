#version 410 core

in vec3 v_view_normal;
out vec3 frag_normal;

void main() {
    frag_normal = normalize(v_view_normal);
}
