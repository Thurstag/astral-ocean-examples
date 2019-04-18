// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <algorithm>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

#include <ao/vulkan/engine/settings.h>
#include <ao/vulkan/wrapper/shader_module.h>
#include <ao/vulkan/buffer/array/basic_buffer.hpp>
#include <ao/vulkan/buffer/tuple/staging_buffer.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vulkan/vulkan.hpp>

#include "../shared/glfw_engine.h"
#include "../shared/ubo.hpp"
#include "../shared/vertex.hpp"

class DepthRectangleDemo : public ao::vulkan::GLFWEngine {
   public:
    std::chrono::time_point<std::chrono::system_clock> clock;
    bool clock_start = false;

    std::vector<Vertex> vertices;
    std::vector<u16> indices;

    std::unique_ptr<ao::vulkan::StagingTupleBuffer<Vertex, u16>> model_buffer;
    std::unique_ptr<ao::vulkan::BasicDynamicArrayBuffer<UniformBufferObject>> ubo_buffer;

    std::vector<UniformBufferObject> uniform_buffers;

    std::vector<ao::vulkan::GraphicsPrimaryCommandBuffer::SecondaryCommandBuffer*> secondary_command_buffers;

    explicit DepthRectangleDemo(std::shared_ptr<ao::vulkan::EngineSettings> settings)
        : ao::vulkan::GLFWEngine(settings),
          vertices({{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
                    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
                    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
                    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}},

                    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},
                    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}}}),
          indices({0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4}){};
    virtual ~DepthRectangleDemo() = default;

    void freeVulkan() override;
    vk::RenderPass createRenderPass() override;
    void createPipelines() override;
    void createVulkanBuffers() override;
    void createSecondaryCommandBuffers() override;
    void beforeCommandBuffersUpdate() override;
};
