#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 rotation;
    mat4 view;
    mat4 proj;
    float scale;
} ubo;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec3 out_position;
layout(location = 2) out vec3 out_color;

void main() {
    out_normal = (ubo.rotation * vec4(in_normal, 1.0)).xyz;
    out_position = (ubo.rotation * vec4(vec3(ubo.scale * in_position), 1.0)).xyz;
    out_color = in_color;

    gl_Position = ubo.proj * ubo.view * vec4(out_position, 1.0);
}