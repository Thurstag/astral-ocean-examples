// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <algorithm>
#include <vector>

#include <ao/vulkan/engine/settings.h>
#include <ao/vulkan/engine/wrappers/shader_module.h>
#include <ao/vulkan/engine/wrappers/buffers/tuple/staging_buffer.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vulkan/vulkan.hpp>

#include "../shared/glfw_engine.h"
#include "../shared/vertex.hpp"

class TriangleDemo : public virtual ao::vulkan::GLFWEngine {
   public:
    std::chrono::time_point<std::chrono::system_clock> clock;
    bool clockInit = false;

    std::vector<Vertex> vertices;
    std::vector<u16> indices;

    std::unique_ptr<ao::vulkan::TupleBuffer<Vertex>> vertexBuffer;
    std::unique_ptr<ao::vulkan::TupleBuffer<u16>> indexBuffer;

    explicit TriangleDemo(std::shared_ptr<ao::vulkan::EngineSettings> settings)
        : ao::vulkan::GLFWEngine(settings),
          ao::vulkan::Engine(settings),
          vertices({{{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}}, {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}}),
          indices({0, 1, 2, 0}){};
    virtual ~TriangleDemo();

    void render() override;
    void setUpRenderPass() override;
    void createPipelineLayouts() override;
    void setUpPipelines() override;
    void setUpVulkanBuffers() override;
    void createSecondaryCommandBuffers() override;
    void executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritanceInfo, int frameIndex, vk::CommandBuffer& primaryCmd) override;
    void updateUniformBuffers() override;
    vk::QueueFlags queueFlags() const override;
    void createDescriptorSetLayouts() override;
    void createDescriptorPools() override;
    void createDescriptorSets() override;
};
