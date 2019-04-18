#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 rotation;
    mat4 view;
    mat4 proj;
    float scale;
} ubo;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_texture_coord;

layout(location = 0) out vec2 out_texture_coord;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.rotation * vec4(vec3(ubo.scale * in_position), 1.0);
    out_texture_coord = in_texture_coord;
}