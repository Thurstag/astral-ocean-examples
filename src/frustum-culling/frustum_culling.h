// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <algorithm>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#define INSTANCE_COUNT 10000

#include <ao/vulkan/engine/settings.h>
#include <ao/vulkan/wrapper/shader_module.h>
#include <ao/vulkan/memory/vector.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vulkan/vulkan.hpp>

#include "../shared/glfw_engine.h"
#include "../shared/ubo.hpp"
#include "../shared/vertex.hpp"

using pixel_t = unsigned char;
using UBO = InstanceUniformBufferObject;
using ComputeCommandBuffer = ao::vulkan::CommandBuffer<>;

namespace ao::vulkan::semaphore {
    static constexpr size_t ComputeProcess = 3;
    static constexpr size_t GraphicProcessAfterCompute = 4;
}  // namespace ao::vulkan::semaphore

class FrustumDemo : public ao::vulkan::GLFWEngine {
   public:
    std::chrono::time_point<std::chrono::system_clock> clock;
    bool clock_start = false;
    int key_last_state;

    std::vector<TexturedVertex> vertices;
    std::vector<u32> indices;
    size_t vertices_count;
    u32 indices_count;

    std::unique_ptr<ao::vulkan::Vector<vk::DrawIndexedIndirectCommand>> draw_command_buffer;
    std::unique_ptr<ao::vulkan::Vector<vk::DispatchIndirectCommand>> dispatch_buffer;
    std::unique_ptr<ao::vulkan::Vector<char>> instance_buffer, model_buffer;
    std::unique_ptr<ao::vulkan::Vector<glm::vec4>> frustum_planes_buffer;
    std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> texture;
    std::unique_ptr<ao::vulkan::Vector<UBO>> ubo_buffer;
    vk::Sampler texture_sampler;

    std::vector<ao::vulkan::GraphicsPrimaryCommandBuffer::SecondaryCommandBuffer*> secondary_command_buffers;

    std::vector<vk::CommandBuffer> primary_compute_command_buffers;
    std::vector<ComputeCommandBuffer*> compute_command_buffers;

    explicit FrustumDemo(std::shared_ptr<ao::vulkan::EngineSettings> settings) : ao::vulkan::GLFWEngine(settings), key_last_state(GLFW_RELEASE){};
    virtual ~FrustumDemo() = default;

    void createSemaphores() override;
    void freeVulkan() override;
    void initVulkan() override;
    void render() override;
    vk::RenderPass createRenderPass() override;
    void createPipelines() override;
    void createVulkanBuffers() override;
    void createSecondaryCommandBuffers() override;
    void beforeCommandBuffersUpdate() override;
};
