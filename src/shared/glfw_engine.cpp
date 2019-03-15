// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "glfw_engine.h"

#include <fstream>

#include <fmt/format.h>
#include <boost/filesystem.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "metrics/counter_metric.hpp"
#include "metrics/duration_metric.hpp"

ao::vulkan::GLFWEngine::GLFWEngine(std::shared_ptr<EngineSettings> settings) : Engine(settings), window(nullptr) {}

void ao::vulkan::GLFWEngine::OnFramebufferSizeCallback(GLFWwindow* window, int width, int height) {
    static_cast<ao::vulkan::GLFWEngine*>(glfwGetWindowUserPointer(window))->enforce_resize = true;
}

void ao::vulkan::GLFWEngine::OnKeyEventCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    static_cast<ao::vulkan::GLFWEngine*>(glfwGetWindowUserPointer(window))->onKeyEventCallback(window, key, scancode, action, mods);
}

void ao::vulkan::GLFWEngine::onKeyEventCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    this->key_states[key].first = this->key_states[key].second;
    this->key_states[key].second = action;
}

std::vector<u8> ao::vulkan::GLFWEngine::LoadCache(std::string const& file) {
    if (!boost::filesystem::exists(file)) {
        return {};
    }

    // Prepare copy
    std::ifstream cache(file);
    std::istream_iterator<u8> start(cache), end;

    return std::vector<u8>(start, end);
}

void ao::vulkan::GLFWEngine::saveCache(std::string const& directory, std::string const& filename, vk::PipelineCache cache) {
    if (!boost::filesystem::exists(directory)) {
        boost::filesystem::create_directories(directory);
    }

    // Get pipeline cache
    auto data = this->device->logical()->getPipelineCacheData(cache);

    // Create file
    std::ofstream output_file(directory + std::string("/") + filename);

    // Copy
    std::ostream_iterator<u8> output_iterator(output_file);
    std::copy(data.begin(), data.end(), output_iterator);
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
    glfwSetWindowUserPointer(this->window, this);

    // Set icon
    std::array<GLFWimage, 1> icons;
    icons.front().pixels = stbi_load("assets/icons/logo.png", &icons.front().width, &icons.front().height, nullptr, STBI_rgb_alpha);
    glfwSetWindowIcon(this->window, static_cast<int>(icons.size()), icons.data());
    stbi_image_free(icons.front().pixels);

    // Define buffer resize callback
    glfwSetFramebufferSizeCallback(this->window, ao::vulkan::GLFWEngine::OnFramebufferSizeCallback);

    // Define key callback
    glfwSetKeyCallback(this->window, ao::vulkan::GLFWEngine::OnKeyEventCallback);
}

vk::SurfaceKHR ao::vulkan::GLFWEngine::createSurface() {
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

    // Free window
    this->freeWindow();

    // Free command buffers
    this->secondary_command_pool.reset();

    // Free wrappers
    for (auto buffer : this->primary_command_buffers) {
        delete buffer;
    }

    ao::vulkan::Engine::freeVulkan();
}

void ao::vulkan::GLFWEngine::initVulkan() {
    ao::vulkan::Engine::initVulkan();

    // Create command pool
    this->secondary_command_pool = std::make_unique<ao::vulkan::CommandPool>(
        this->device->logical(), vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        this->device->queues()->at(vk::to_string(vk::QueueFlagBits::eGraphics)).family_index, ao::vulkan::CommandPoolAccessMode::eConcurrent);

    // Init metric module (TODO: DrawCall per second)
    this->metrics = std::make_unique<ao::vulkan::MetricModule>(this->device);
    this->metrics->add("CPU", new ao::vulkan::BasicDurationMetric<std::chrono::duration<double, std::milli>>("ms"));
    this->metrics->add(
        "GPU", new ao::vulkan::DurationCommandBufferMetric<std::milli>("ms", std::make_pair(this->device, this->metrics->timestampQueryPool())));
    this->metrics->add("Triangle/s", new ao::vulkan::CounterCommandBufferMetric<std::chrono::seconds, u64>(
                                         0, std::make_pair(this->device, this->metrics->triangleQueryPool())));
    this->metrics->add("Frame/s", new ao::vulkan::CounterMetric<std::chrono::seconds, int>(0));
}

void ao::vulkan::GLFWEngine::prepareVulkan() {
    ao::vulkan::Engine::prepareVulkan();

    // Create primary wrappers
    this->primary_command_buffers.resize(this->swapchain->size());
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        this->primary_command_buffers[i] = new ao::vulkan::PrimaryCommandBuffer(
            this->swapchain->commandBuffers()[i], ao::vulkan::ExecutionPolicy::eSequenced,
            [& metrics = this->metrics](vk::CommandBuffer command_buffer, vk::ArrayProxy<vk::CommandBuffer const> secondary_command_buffers,
                                        vk::RenderPass render_pass, vk::Framebuffer frame, vk::Extent2D swapchain_extent, int frame_index) {
                // Clear values
                std::array<vk::ClearValue, 2> clear_values;
                clear_values[0].setColor(vk::ClearColorValue());
                clear_values[1].setDepthStencil(vk::ClearDepthStencilValue(1));

                // Begin info
                vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eRenderPassContinue);

                // Render pass info
                vk::RenderPassBeginInfo render_pass_info(render_pass, frame, vk::Rect2D().setExtent(swapchain_extent),
                                                         static_cast<u32>(clear_values.size()), clear_values.data());

                command_buffer.begin(begin_info);
                {
                    // Reset pools
                    command_buffer.resetQueryPool(metrics->timestampQueryPool(), frame_index * 2, 2);
                    command_buffer.resetQueryPool(metrics->triangleQueryPool(), frame_index * 4, 4);

                    // Statistics
                    command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, metrics->timestampQueryPool(), frame_index * 2);
                    command_buffer.beginQuery(metrics->triangleQueryPool(), frame_index * 4, vk::QueryControlFlags());

                    // Execute secondary command buffers
                    command_buffer.beginRenderPass(render_pass_info, vk::SubpassContents::eSecondaryCommandBuffers);
                    {
                        if (!secondary_command_buffers.empty()) {
                            command_buffer.executeCommands(secondary_command_buffers);
                        }
                    }
                    command_buffer.endRenderPass();

                    // Statistics
                    command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, metrics->timestampQueryPool(), (frame_index * 2) + 1);
                    command_buffer.endQuery(metrics->triangleQueryPool(), frame_index * 4);
                }
                command_buffer.end();
            },
            [](vk::RenderPass render_pass, vk::Framebuffer frame) {
                return vk::CommandBufferInheritanceInfo(render_pass, 0, frame)
                    .setPipelineStatistics(vk::QueryPipelineStatisticFlagBits::eClippingInvocations);
            });
    }

    // Create secondary command buffers
    this->createSecondaryCommandBuffers();
}

void ao::vulkan::GLFWEngine::render() {
    auto cpuFrame = static_cast<ao::vulkan::DurationMetric*>((*this->metrics)["CPU"]);
    auto fps = static_cast<ao::vulkan::CounterMetric<std::chrono::seconds, int>*>((*this->metrics)["Frame/s"]);
    auto triangle_count = static_cast<ao::vulkan::CounterCommandBufferMetric<std::chrono::seconds, u64>*>((*this->metrics)["Triangle/s"]);

    // Render
    cpuFrame->start();
    ao::vulkan::Engine::render();
    cpuFrame->stop();
    fps->increment();
    triangle_count->update();

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

std::vector<vk::PhysicalDeviceFeatures> ao::vulkan::GLFWEngine::deviceFeatures() const {
    return {vk::PhysicalDeviceFeatures().setSamplerAnisotropy(VK_TRUE)};
}

std::vector<char const*> ao::vulkan::GLFWEngine::instanceExtensions() const {
    return ao::vulkan::utilities::extensions();
}

void ao::vulkan::GLFWEngine::updateCommandBuffers() {
    // Invalidate all command buffers
    if (this->swapchain->state() == ao::vulkan::SwapchainState::eReset) {
        for (auto buffer : this->primary_command_buffers) {
            buffer->invalidate();
        }
    }

    auto index = this->swapchain->frameIndex();

    // Update command buffer
    if (this->primary_command_buffers[index]->state() == ao::vulkan::CommandBufferState::eOutdate) {
        this->primary_command_buffers[index]->update(this->render_pass, this->swapchain->currentFrame(), this->swapchain->extent(), index);
    }
}

void ao::vulkan::GLFWEngine::afterFrame() {}

std::vector<ao::vulkan::QueueRequest> ao::vulkan::GLFWEngine::requestQueues() const {
    auto families = this->device->physical().getQueueFamilyProperties();

    // Get indices
    auto transfer_index = ao::vulkan::utilities::findQueueFamilyIndex(families, vk::QueueFlagBits::eTransfer);
    auto graphics_index = ao::vulkan::utilities::findQueueFamilyIndex(families, vk::QueueFlagBits::eGraphics);
    auto compute_index = ao::vulkan::utilities::findQueueFamilyIndex(families, vk::QueueFlagBits::eCompute);

    if (compute_index == graphics_index) {
        return {ao::vulkan::QueueRequest(vk::QueueFlagBits::eGraphics, 1, (families[graphics_index].queueCount - 2) / 2),
                ao::vulkan::QueueRequest(vk::QueueFlagBits::eTransfer, 0, families[transfer_index].queueCount),
                ao::vulkan::QueueRequest(vk::QueueFlagBits::eCompute, 1, (families[compute_index].queueCount - 2) / 2)};
    }
    return {ao::vulkan::QueueRequest(vk::QueueFlagBits::eGraphics, 1, families[graphics_index].queueCount - 1),
            ao::vulkan::QueueRequest(vk::QueueFlagBits::eTransfer, 0, families[transfer_index].queueCount),
            ao::vulkan::QueueRequest(vk::QueueFlagBits::eCompute, 1, families[compute_index].queueCount - 1)};
}
