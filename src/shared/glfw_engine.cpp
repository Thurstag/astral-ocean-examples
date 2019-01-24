// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "glfw_engine.h"

#include "metrics/counter_metric.hpp"
#include "metrics/duration_metric.hpp"

/// <summary>
/// Destructor
/// </summary>
ao::vulkan::GLFWEngine::~GLFWEngine() {
    this->freeWindow();
}

void ao::vulkan::GLFWEngine::initWindow() {
    glfwInit();

    // Define properties
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, this->settings_.window.rezisable ? GLFW_TRUE : GLFW_FALSE);

    // Create window
    this->window = glfwCreateWindow(static_cast<int>(this->settings_.window.width), static_cast<int>(this->settings_.window.height),
                                    this->settings_.window.name.c_str(), nullptr, nullptr);
}

void ao::vulkan::GLFWEngine::initSurface(vk::SurfaceKHR& surface) {
    VkSurfaceKHR _s;
    ao::vulkan::utilities::vkAssert(glfwCreateWindowSurface(*this->instance, this->window, nullptr, &_s), "Fail to create surface");

    surface = _s;
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

    // Init metric module
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
        glfwSetWindowTitle(this->window, this->metrics->toString().c_str());

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
    vk::CommandBuffer& currentCommand = this->swapchain->commands["primary"].buffers[this->frameBufferIndex];
    vk::Framebuffer& currentFrame = this->frames[this->frameBufferIndex];
    auto& helpers = this->swapchain->command_helpers;
    int index = this->frameBufferIndex;
    std::mutex cmdMutex;

    // Create info
    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue);

    vk::RenderPassBeginInfo renderPassInfo(this->renderPass, currentFrame, this->swapchain->command_helpers.second,
                                           static_cast<u32>(this->swapchain->command_helpers.first.size()),
                                           this->swapchain->command_helpers.first.data());

    currentCommand.begin(&beginInfo);
    currentCommand.resetQueryPool(this->metrics->queryPool(), 0, 2);
    currentCommand.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, this->metrics->queryPool(), 0);
    currentCommand.beginRenderPass(renderPassInfo, vk::SubpassContents::eSecondaryCommandBuffers);
    {
        // Create inheritance info for the secondary command buffers
        vk::CommandBufferInheritanceInfo inheritanceInfo(this->renderPass, 0, currentFrame);

        // Get functions
        std::vector<ao::vulkan::DrawInCommandBuffer> functions = this->updateSecondaryCommandBuffers();

        // Execute drawing functions
        std::for_each(std::execution::par, functions.begin(), functions.end(),
                      [index, &inheritanceInfo, &helpers, &cmdMutex, &currentCommand](ao::vulkan::DrawInCommandBuffer& function) {
                          auto& cmd = function(index, inheritanceInfo, helpers);

                          // Add to primary
                          cmdMutex.lock();
                          currentCommand.executeCommands(cmd);
                          cmdMutex.unlock();
                      });
    }
    currentCommand.endRenderPass();
    currentCommand.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, this->metrics->queryPool(), 1);
    currentCommand.end();
}
