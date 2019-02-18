#version 450
#extension GL_ARB_separate_shader_objects : enable

struct InstanceData {
    mat4 rotation;
    vec4 positionAndScale;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    InstanceData instances[2];
} ubo;

layout(push_constant) uniform PushConsts {
	uint array_layer_index;
} pushConsts;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec3 fragTexCoord;

void main() {
    vec4 position = ubo.instances[gl_InstanceIndex].rotation * vec4(vec3(ubo.instances[gl_InstanceIndex].positionAndScale.w * vec3(inPosition, 0.0)), 1.0);
    position += vec4(ubo.instances[gl_InstanceIndex].positionAndScale.xyz, 0.0);

    gl_Position = ubo.proj * ubo.view * position;
    fragTexCoord = vec3(inTexCoord, gl_InstanceIndex + pushConsts.array_layer_index);
}