// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <algorithm>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL

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

class MipmapDemo : public ao::vulkan::GLFWEngine {
   public:
    std::tuple<int, int, int, int, int, int> direction_last_states;
    std::chrono::time_point<std::chrono::system_clock> clock;
    bool engine_start = false;
    int toggle_last_state;

    std::vector<TexturedVertex> vertices;
    std::vector<u32> indices;
    size_t vertices_count;
    u32 indices_count;

    std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> texture, mipmap_texture;
    std::unique_ptr<ao::vulkan::Vector<UniformBufferObject>> ubo_buffer;
    std::unique_ptr<ao::vulkan::Vector<char>> model_buffer;
    vk::Sampler texture_sampler, mipmap_texture_sampler;

    std::vector<ao::vulkan::GraphicsPrimaryCommandBuffer::SecondaryCommandBuffer*> secondary_command_buffers;

    std::tuple<glm::vec3, float, float, float> camera;

    explicit MipmapDemo(std::shared_ptr<ao::vulkan::EngineSettings> settings) : ao::vulkan::GLFWEngine(settings){};
    virtual ~MipmapDemo() = default;

    /**
     * @brief Set-up texture stuff
     *
     */
    void setUpTexture();

    void initVulkan() override;
    void freeVulkan() override;
    vk::RenderPass createRenderPass() override;
    void createPipelines() override;
    void createVulkanBuffers() override;
    void createSecondaryCommandBuffers() override;
    void beforeCommandBuffersUpdate() override;
};
