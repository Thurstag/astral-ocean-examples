// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;

    static std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributes;

        attributes[0] = vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos));
        attributes[1] = vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color));
        return attributes;
    }
};
