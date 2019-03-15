#version 450
#extension GL_ARB_separate_shader_objects : enable

struct InstanceData {
    mat4 rotation;
    vec4 position_and_scale;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    InstanceData instances[4096];
} ubo;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 out_color;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.instances[gl_InstanceIndex].rotation * vec4(vec3(ubo.instances[gl_InstanceIndex].position_and_scale.w * in_position), 1.0);
    out_color = in_color;
}