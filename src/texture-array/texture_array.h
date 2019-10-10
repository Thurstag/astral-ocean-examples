// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <algorithm>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#define INSTANCE_COUNT 3

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

class TextureArrayDemo : public ao::vulkan::GLFWEngine {
   public:
    std::chrono::time_point<std::chrono::system_clock> clock;
    std::pair<int, int> key_last_states;
    bool clock_start = false;

    std::vector<TexturedVertex> vertices;
    std::vector<u16> indices;

    std::unique_ptr<ao::vulkan::Vector<UBO::InstanceData>> instance_buffer;
    std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> texture;
    std::unique_ptr<ao::vulkan::Vector<char>> model_buffer;
    std::unique_ptr<ao::vulkan::Vector<UBO>> ubo_buffer;
    vk::Sampler texture_sampler;

    std::vector<ao::vulkan::GraphicsPrimaryCommandBuffer::SecondaryCommandBuffer*> secondary_command_buffers;

    u32 array_level_index = 0;
    u32 array_levels;

    explicit TextureArrayDemo(std::shared_ptr<ao::vulkan::EngineSettings> settings)
        : ao::vulkan::GLFWEngine(settings),
          vertices({{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f}},
                    {{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f}},
                    {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f}},
                    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f}}}),
          indices({0, 1, 2, 2, 3, 0}){};
    virtual ~TextureArrayDemo() = default;

    void initVulkan() override;
    void freeVulkan() override;
    vk::RenderPass createRenderPass() override;
    void createPipelines() override;
    void createVulkanBuffers() override;
    void createSecondaryCommandBuffers() override;
    void beforeCommandBuffersUpdate() override;
};
