#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 rotation;
    mat4 view;
    mat4 proj;
    float scale;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.rotation * vec4(vec3(ubo.scale * inPosition), 1.0);
    fragTexCoord = inTexCoord;
}