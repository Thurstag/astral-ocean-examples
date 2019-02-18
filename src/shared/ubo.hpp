// Copyright 2018 Astral-Ocean Project
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

template<size_t InstanceCount = 1>
struct InstanceUniformBufferObject {
    struct InstanceData {
        glm::mat4 rotation;
        glm::vec4 positionAndScale;  // x,y,z = position & w = scale
    };

    glm::mat4 view;
    glm::mat4 proj;
    InstanceData instances[InstanceCount];
};