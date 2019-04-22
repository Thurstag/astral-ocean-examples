#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texture_sampler;
layout(binding = 2) uniform sampler2D mipmap_texture_sampler;

layout(push_constant) uniform PushConsts {
	uint sampler_index;
} push_constants;

layout(location = 0) in vec2 in_texture_coord;

layout(location = 0) out vec4 out_color;

void main() {
    if (push_constants.sampler_index > 0) {
        out_color = texture(mipmap_texture_sampler, in_texture_coord);
    } else {
        out_color = texture(texture_sampler, in_texture_coord);
    }
}