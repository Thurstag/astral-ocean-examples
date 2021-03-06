#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConsts {
	uint array_layer_index;
} push_constants;

// Vertex data
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texture_coord;

// Instance data
layout(location = 2) in mat4 in_instance_rotation;
layout(location = 6) in vec4 in_instance_position_and_scale;

layout(location = 0) out vec3 out_texture_coord;

void main() {
    gl_Position = ubo.proj * ubo.view * vec4((in_instance_rotation * vec4(vec2(in_instance_position_and_scale.w * in_position), 0.0, 1.0)).xyz + in_instance_position_and_scale.xyz, 1.0);
    out_texture_coord = vec3(in_texture_coord, gl_InstanceIndex + push_constants.array_layer_index);
}