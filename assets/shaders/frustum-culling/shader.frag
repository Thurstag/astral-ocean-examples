#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texture_sampler;

layout(location = 0) in vec2 in_texture_coord;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(texture_sampler, in_texture_coord);
}