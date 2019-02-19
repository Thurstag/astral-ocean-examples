// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <algorithm>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#define INSTANCE_COUNT 3

#include <ao/vulkan/utilities/settings.h>
#include <ao/vulkan/wrapper/shader_module.h>
#include <ao/vulkan/wrapper/buffer/array/basic_buffer.hpp>
#include <ao/vulkan/wrapper/buffer/tuple/staging_buffer.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vulkan/vulkan.hpp>

#include "../shared/glfw_engine.h"
#include "../shared/ubo.hpp"
#include "../shared/vertex.hpp"

using pixel_t = unsigned char;
using UBO = InstanceUniformBufferObject<INSTANCE_COUNT>;

class TextureArrayDemo : public virtual ao::vulkan::GLFWEngine {
   public:
    std::chrono::time_point<std::chrono::system_clock> clock;
    bool clock_start = false;

    std::vector<TexturedVertex> vertices;
    std::vector<u16> indices;

    std::unique_ptr<ao::vulkan::BasicDynamicArrayBuffer<UBO>> ubo_buffer;
    std::unique_ptr<ao::vulkan::StagingTupleBuffer<TexturedVertex, u16>> model_buffer;
    std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> texture;
    vk::Sampler texture_sampler;

    std::vector<UBO> uniform_buffers;

    std::map<vk::CommandBuffer, bool> to_update;

    u32 array_level_index = 0;
    u32 array_levels;

    explicit TextureArrayDemo(std::shared_ptr<ao::vulkan::EngineSettings> settings)
        : ao::vulkan::GLFWEngine(settings),
          ao::vulkan::Engine(settings),
          vertices({{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f}},
                    {{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f}},
                    {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f}},
                    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f}}}),
          indices({0, 1, 2, 2, 3, 0}){};
    virtual ~TextureArrayDemo() = default;

    virtual void onKeyEventCallback(GLFWwindow* window, int key, int scancode, int action, int mods) override;
    void freeVulkan() override;
    vk::RenderPass createRenderPass() override;
    void createPipelines() override;
    void createVulkanBuffers() override;
    void createSecondaryCommandBuffers() override;
    void executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritance_info, int frame_index,
                                        vk::CommandBuffer primary_command) override;
    void beforeCommandBuffersUpdate() override;
};
