// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <glm/glm.hpp>

struct UniformBufferObject {
    glm::mat4 rotation;
    glm::mat4 view;
    glm::mat4 proj;
    float scale = 1.0f;
};

struct UniformBufferLightObject {
    glm::vec4 view_position;
    glm::vec4 position;
    glm::vec4 color;  // x,y,z = color & w = ambient strength
};

struct InstanceUniformBufferObject {
    struct InstanceData {
        glm::mat4 rotation;
        glm::vec4 position_and_scale;  // x,y,z = position & w = scale
    };

    glm::mat4 view;
    glm::mat4 proj;
};