// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <algorithm>
#include <vector>

#include <ao/vulkan/engine/settings.h>
#include <ao/vulkan/wrapper/shader_module.h>
#include <ao/vulkan/buffer/tuple/staging_buffer.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vulkan/vulkan.hpp>

#include "../shared/glfw_engine.h"
#include "../shared/vertex.hpp"

class TriangleDemo : public virtual ao::vulkan::GLFWEngine {
   public:
    std::chrono::time_point<std::chrono::system_clock> clock;
    bool clock_start = false;

    std::vector<Vertex> vertices;
    std::vector<u16> indices;

    std::unique_ptr<ao::vulkan::StagingTupleBuffer<Vertex>> vertices_buffer;
    std::unique_ptr<ao::vulkan::StagingTupleBuffer<u16>> indices_buffer;

    std::map<vk::CommandBuffer, bool> to_update;

    explicit TriangleDemo(std::shared_ptr<ao::vulkan::EngineSettings> settings)
        : ao::vulkan::GLFWEngine(settings),
          ao::vulkan::Engine(settings),
          vertices({{{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}}, {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}}),
          indices({0, 1, 2, 0}){};
    virtual ~TriangleDemo() = default;

    void freeVulkan() override;
    vk::RenderPass createRenderPass() override;
    void createPipelines() override;
    void createVulkanBuffers() override;
    void createSecondaryCommandBuffers() override;
    void executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritance_info, int frame_index,
                                        vk::CommandBuffer primary_command) override;
    void beforeCommandBuffersUpdate() override;
};
