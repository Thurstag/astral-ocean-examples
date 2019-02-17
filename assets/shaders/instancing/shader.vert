#version 450
#extension GL_ARB_separate_shader_objects : enable

struct InstanceData {
    mat4 position;
    mat4 rotation;
    float scale;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    InstanceData instances[4096];
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.instances[gl_InstanceIndex].rotation * vec4(vec3(ubo.instances[gl_InstanceIndex].scale * inPosition), 1.0);
    fragColor = inColor;
}