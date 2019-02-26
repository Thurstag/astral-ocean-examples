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

using pixel_t = unsigned char;

class MipmapDemo : public virtual ao::vulkan::GLFWEngine {
   public:
    std::chrono::time_point<std::chrono::system_clock> clock;
    bool clock_start = false;

    std::vector<TexturedVertex> vertices;
    std::vector<u32> indices;
    u32 indices_count;

    std::unique_ptr<ao::vulkan::BasicDynamicArrayBuffer<UniformBufferObject>> ubo_buffer;
    std::unique_ptr<ao::vulkan::StagingTupleBuffer<TexturedVertex, u32>> model_buffer;
    std::tuple<vk::Image, vk::DeviceMemory, vk::ImageView> texture;
    vk::Sampler texture_sampler;

    std::vector<UniformBufferObject> uniform_buffers;

    std::map<vk::CommandBuffer, bool> to_update;

    std::tuple<glm::vec3, float, float, float> camera;

    explicit MipmapDemo(std::shared_ptr<ao::vulkan::EngineSettings> settings) : ao::vulkan::GLFWEngine(settings), ao::vulkan::Engine(settings){};
    virtual ~MipmapDemo() = default;

    /**
     * @brief Set-up texture stuff
     *
     */
    void setUpTexture();

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
