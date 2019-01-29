// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "glfw_engine.h"

#include <fmt/format.h>

#include "metrics/counter_metric.hpp"
#include "metrics/duration_metric.hpp"

ao::vulkan::GLFWEngine::GLFWEngine(std::shared_ptr<EngineSettings> settings) : Engine(settings), window(nullptr) {}

/// <summary>
/// Destructor
/// </summary>
ao::vulkan::GLFWEngine::~GLFWEngine() {
    this->freeWindow();

    this->secondary_command_pool.reset();
}

void ao::vulkan::GLFWEngine::initWindow() {
    glfwInit();

    // Define properties
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,
                   this->settings_->get<bool>(ao::vulkan::settings::WindowResizable, std::make_optional<bool>(true)) ? GLFW_TRUE : GLFW_FALSE);

    if (!this->settings_->exists(ao::vulkan::settings::WindowTitle)) {
        this->settings_->get<std::string>(ao::vulkan::settings::WindowTitle) = std::string("Undefined");
    }

    // Create window
    this->window = glfwCreateWindow(static_cast<int>(this->settings_->get<u64>(ao::vulkan::settings::WindowWidth)),
                                    static_cast<int>(this->settings_->get<u64>(ao::vulkan::settings::WindowHeight)),
                                    this->settings_->get<std::string>(ao::vulkan::settings::WindowTitle).c_str(), nullptr, nullptr);
}

vk::SurfaceKHR ao::vulkan::GLFWEngine::initSurface() {
    VkSurfaceKHR _s;
    ao::vulkan::utilities::vkAssert(glfwCreateWindowSurface(*this->instance, this->window, nullptr, &_s), "Fail to create surface");

    return vk::SurfaceKHR(_s);
}

void ao::vulkan::GLFWEngine::freeWindow() {
    glfwDestroyWindow(this->window);
    glfwTerminate();
}

bool ao::vulkan::GLFWEngine::isIconified() const {
    return glfwGetWindowAttrib(this->window, GLFW_ICONIFIED);
}

void ao::vulkan::GLFWEngine::freeVulkan() {
    // Free module
    this->metrics.reset();

    ao::vulkan::Engine::freeVulkan();
}

void ao::vulkan::GLFWEngine::initVulkan() {
    ao::vulkan::Engine::initVulkan();

    // Create command pool
    this->secondary_command_pool = std::make_unique<ao::vulkan::CommandPool>(
        this->device->logical, vk::CommandPoolCreateFlagBits::eResetCommandBuffer, this->device->queues[vk::QueueFlagBits::eGraphics].index,
        ao::vulkan::CommandPoolAccessModeFlagBits::eConcurrent);

    // Init metric module (TODO: DrawCall per second)
    this->metrics = std::make_unique<ao::vulkan::MetricModule>(this->device);
    this->metrics->add("CPU", new ao::vulkan::BasicDurationMetric<std::chrono::duration<double, std::milli>>("ms"));
    this->metrics->add("GPU", new ao::vulkan::CommandBufferMetric<std::milli>(
                                  "ms", std::make_pair(std::weak_ptr<ao::vulkan::Device>(this->device), this->metrics->queryPool())));
    this->metrics->add("FPS", new ao::vulkan::CounterMetric<std::chrono::seconds, int>(0));
}

void ao::vulkan::GLFWEngine::render() {
    auto cpuFrame = static_cast<ao::vulkan::DurationMetric*>((*this->metrics)["CPU"]);
    auto fps = static_cast<ao::vulkan::CounterMetric<std::chrono::seconds, int>*>((*this->metrics)["FPS"]);

    // Render
    cpuFrame->start();
    ao::vulkan::Engine::render();
    cpuFrame->stop();
    fps->increment();

    // Poll events after rendering
    glfwPollEvents();

    // Display metrics
    if (fps->hasToBeReset()) {
        glfwSetWindowTitle(
            this->window, fmt::format("{} [{}]", this->settings_->get<std::string>(ao::vulkan::settings::WindowTitle), this->metrics->str()).c_str());

        // Reset metrics
        this->metrics->reset();
    }
}

bool ao::vulkan::GLFWEngine::loopingCondition() const {
    return !glfwWindowShouldClose(this->window);
}

void ao::vulkan::GLFWEngine::waitMaximized() {
    glfwWaitEvents();
}

std::vector<char const*> ao::vulkan::GLFWEngine::instanceExtensions() const {
    return ao::vulkan::utilities::extensions();
}

void ao::vulkan::GLFWEngine::updateCommandBuffers() {
    // Get current command buffer/frame
    vk::CommandBuffer command = this->swapchain->currentCommand();
    vk::Framebuffer frame = this->swapchain->currentFrame();

    // Create info
    vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eRenderPassContinue);

    std::array<vk::ClearValue, 2> clearValues;
    clearValues[0].setColor(vk::ClearColorValue());
    clearValues[1].setDepthStencil(vk::ClearDepthStencilValue(1));

    vk::RenderPassBeginInfo render_pass_info(this->renderPass, frame, vk::Rect2D().setExtent(this->swapchain->current_extent),
                                             static_cast<u32>(clearValues.size()), clearValues.data());

    command.begin(&begin_info);
    command.resetQueryPool(this->metrics->queryPool(), 0, 2);
    command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, this->metrics->queryPool(), 0);
    command.beginRenderPass(render_pass_info, vk::SubpassContents::eSecondaryCommandBuffers);
    {
        // Create inheritance info for the secondary command buffers
        vk::CommandBufferInheritanceInfo inheritanceInfo(this->renderPass, 0, frame);

        // Execute secondary command buffers
        this->executeSecondaryCommandBuffers(inheritanceInfo, this->swapchain->frame_index, command);
    }
    command.endRenderPass();
    command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, this->metrics->queryPool(), 1);
    command.end();
}

void ao::vulkan::GLFWEngine::afterFrame() {}
