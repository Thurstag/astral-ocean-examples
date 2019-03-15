#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform UniformBufferLightObject {
    vec4 view_position;
    vec4 position;
    vec4 color;
} ubo;

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_position;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec4 out_color;

void main() {
    // Ambient
    vec3 ambient = ubo.color.w * ubo.color.xyz;
  	
    // Diffuse 
    vec3 _in_normal = normalize(in_normal);
    vec3 light_direction = normalize(ubo.position.xyz - in_position);
    vec3 diffuse = max(dot(_in_normal, light_direction), 0.0) * ubo.color.xyz;

    // Specular
    vec3 view_direction = normalize(ubo.view_position.xyz - in_position);
    vec3 reflect_direction = reflect(-light_direction, _in_normal);
    vec3 specular = 0.5 * pow(max(dot(view_direction, reflect_direction), 0.0), 32) * ubo.color.xyz;

    out_color = vec4(vec3((ambient + diffuse + specular) * in_color), 1.0);
}