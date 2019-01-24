// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <algorithm>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

#include <ao/vulkan/engine/settings.h>
#include <ao/vulkan/engine/wrappers/shadermodule.h>
#include <ao/vulkan/engine/wrappers/buffers/array/basic_buffer.hpp>
#include <ao/vulkan/engine/wrappers/buffers/tuple/staging_buffer.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vulkan/vulkan.hpp>

#include "../shared/glfw_engine.h"
#include "../shared/ubo.hpp"
#include "../shared/vertex.hpp"

using pixel_t = unsigned char;

class TexturedRectangle : public virtual ao::vulkan::GLFWEngine {
   public:
    std::chrono::time_point<std::chrono::system_clock> clock;
    bool clockInit = false;

    std::vector<Vertex> vertices;
    std::vector<u16> indices;

    std::unique_ptr<ao::vulkan::TupleBuffer<Vertex, u16>> rectangleBuffer;
    std::unique_ptr<ao::vulkan::DynamicArrayBuffer<UniformBufferObject>> uniformBuffer;
    std::unique_ptr<ao::vulkan::StagingTupleBuffer<pixel_t>> textureBuffer;

    std::vector<UniformBufferObject> _uniformBuffers;

    explicit TexturedRectangle(ao::vulkan::EngineSettings settings)
        : ao::vulkan::GLFWEngine(settings),
          ao::vulkan::Engine(settings),
          vertices({{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}}),
          indices({0, 1, 2, 2, 3, 0}){};
    virtual ~TexturedRectangle();

    void setUpRenderPass() override;
    void createPipelineLayouts() override;
    void setUpPipelines() override;
    void setUpVulkanBuffers() override;
    void createSecondaryCommandBuffers() override;
    std::vector<ao::vulkan::DrawInCommandBuffer> updateSecondaryCommandBuffers() override;
    void updateUniformBuffers() override;
    vk::QueueFlags queueFlags() const override;
    void createDescriptorSetLayouts() override;
    void createDescriptorPools() override;
    void createDescriptorSets() override;
};
