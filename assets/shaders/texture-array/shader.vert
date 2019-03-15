#version 450
#extension GL_ARB_separate_shader_objects : enable

struct InstanceData {
    mat4 rotation;
    vec4 position_and_scale;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    InstanceData instances[2];
} ubo;

layout(push_constant) uniform PushConsts {
	uint array_layer_index;
} push_constants;

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texture_coord;

layout(location = 0) out vec3 out_texture_coord;

void main() {
    vec4 position = ubo.instances[gl_InstanceIndex].rotation * vec4(vec3(ubo.instances[gl_InstanceIndex].position_and_scale.w * vec3(in_position, 0.0)), 1.0);
    position += vec4(ubo.instances[gl_InstanceIndex].position_and_scale.xyz, 0.0);

    gl_Position = ubo.proj * ubo.view * position;
    out_texture_coord = vec3(in_texture_coord, gl_InstanceIndex + push_constants.array_layer_index);
}