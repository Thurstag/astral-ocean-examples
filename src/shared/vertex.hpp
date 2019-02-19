// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <ao/core/utilities/types.h>
#include <vulkan/vulkan.hpp>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;

    static std::array<vk::VertexInputAttributeDescription, 2> AttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributes;

        attributes[0] = vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos));
        attributes[1] = vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color));
        return attributes;
    }
};

struct TexturedVertex {
    glm::vec3 pos;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription BindingDescription() {
        return vk::VertexInputBindingDescription(0, sizeof(TexturedVertex), vk::VertexInputRate::eVertex);
    }

    static std::array<vk::VertexInputAttributeDescription, 2> AttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributes;

        attributes[0] = vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(TexturedVertex, pos));
        attributes[1] = vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(TexturedVertex, texCoord));
        return attributes;
    }
};

struct MeshOptVertex {
    float px, py, pz;
    float nx, ny, nz;
    float tx, ty;
};
