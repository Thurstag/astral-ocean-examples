#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_position;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(in_color, 1);
}