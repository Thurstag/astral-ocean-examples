// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "frustum_culling.h"

#include <ao/vulkan/pipeline/compute_pipeline.h>
#include <ao/vulkan/pipeline/graphics_pipeline.h>
#include <meshoptimizer.h>
#include <objparser.h>
#include <stb_image.h>
#include <boost/filesystem.hpp>

#include "../shared/metrics/counter_metric.hpp"
#include "../shared/metrics/duration_metric.hpp"

static constexpr char const* CullingEnableKey = "culling.enable";

void FrustumDemo::onKeyEventCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    ao::vulkan::GLFWEngine::onKeyEventCallback(window, key, scancode, action, mods);

    // Toggle culling
    if (key == GLFW_KEY_T && action == GLFW_PRESS) {
        this->settings_->get<bool>(CullingEnableKey) = !this->settings_->get<bool>(CullingEnableKey);

        // Update draw commands/dispatch buffers
        auto draw_commands =
            std::vector<vk::DrawIndexedIndirectCommand>(this->swapchain->size(), vk::DrawIndexedIndirectCommand(this->indices_count, INSTANCE_COUNT));
        this->draw_command_buffer->update(draw_commands);

        // Toggle gpu(compute) metric
        if (this->settings_->get<bool>(CullingEnableKey)) {
            this->metrics->add("GPU(Compute)", new ao::vulkan::DurationCommandBufferMetric<std::milli, 2>(
                                                   "ms", std::make_pair(this->device, this->metrics->timestampQueryPool())));
        } else {
            this->metrics->remove("GPU(Compute)");
        }

        this->LOGGER << ao::core::Logger::Level::debug
                     << fmt::format("Frustum culling: {}", this->settings_->get<bool>(CullingEnableKey) ? "On" : "Off");
    }
}

void FrustumDemo::createSemaphores() {
    this->semaphores = ao::vulkan::SemaphoreContainer(this->device->logical());

    // Create semaphores
    this->semaphores.resize(5 * this->swapchain->size());
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        vk::Semaphore acquire = this->device->logical()->createSemaphore(vk::SemaphoreCreateInfo());
        vk::Semaphore render = this->device->logical()->createSemaphore(vk::SemaphoreCreateInfo());
        vk::Semaphore compute = this->device->logical()->createSemaphore(vk::SemaphoreCreateInfo());

        // Fill container
        this->semaphores[(ao::vulkan::semaphore::AcquireImage * this->swapchain->size()) + i].signals.push_back(acquire);

        this->semaphores[(ao::vulkan::semaphore::ComputeProcess * this->swapchain->size()) + i].signals.push_back(compute);

        this->semaphores[(ao::vulkan::semaphore::GraphicProcessAfterCompute * this->swapchain->size()) + i].waits.push_back(acquire);
        this->semaphores[(ao::vulkan::semaphore::GraphicProcessAfterCompute * this->swapchain->size()) + i].waits.push_back(compute);
        this->semaphores[(ao::vulkan::semaphore::GraphicProcessAfterCompute * this->swapchain->size()) + i].signals.push_back(render);

        this->semaphores[(ao::vulkan::semaphore::GraphicProcess * this->swapchain->size()) + i].waits.push_back(acquire);
        this->semaphores[(ao::vulkan::semaphore::GraphicProcess * this->swapchain->size()) + i].signals.push_back(render);

        this->semaphores[(ao::vulkan::semaphore::PresentImage * this->swapchain->size()) + i].waits.push_back(render);
    }
}

void FrustumDemo::freeVulkan() {
    // Free buffers
    this->frustum_planes_buffer.reset();
    this->draw_command_buffer.reset();
    this->model_buffer.reset();
    this->ubo_buffer.reset();
    this->instance_buffer.reset();
    this->dispatch_buffer.reset();

    // Free wrappers
    for (auto buffer : this->secondary_command_buffers) {
        delete buffer;
    }
    for (auto buffer : this->compute_command_buffers) {
        delete buffer;
    }

    this->device->logical()->destroySampler(this->texture_sampler);

    this->device->logical()->destroyImage(std::get<0>(this->texture));
    this->device->logical()->destroyImageView(std::get<2>(this->texture));
    this->device->logical()->freeMemory(std::get<1>(this->texture));

    ao::vulkan::GLFWEngine::freeVulkan();
}

void FrustumDemo::initVulkan() {
    ao::vulkan::Engine::initVulkan();

    // Create command pool
    this->secondary_command_pool = std::make_unique<ao::vulkan::CommandPool>(
        this->device->logical(), vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        this->device->queues()->at(vk::to_string(vk::QueueFlagBits::eGraphics)).family_index, ao::vulkan::CommandPoolAccessMode::eConcurrent);

    // Init metric module
    this->metrics = std::make_unique<ao::vulkan::MetricModule>(this->device);
    this->metrics->add("CPU", new ao::vulkan::BasicDurationMetric<std::chrono::duration<double, std::milli>>("ms"));
    this->metrics->add("GPU(Graphics)", new ao::vulkan::DurationCommandBufferMetric<std::milli>(
                                            "ms", std::make_pair(this->device, this->metrics->timestampQueryPool())));
    this->metrics->add("GPU(Compute)", new ao::vulkan::DurationCommandBufferMetric<std::milli, 2>(
                                           "ms", std::make_pair(this->device, this->metrics->timestampQueryPool())));
    this->metrics->add("Triangle/s", new ao::vulkan::CounterCommandBufferMetric<std::chrono::seconds, u64>(
                                         0, std::make_pair(this->device, this->metrics->triangleQueryPool())));
    this->metrics->add("Frame/s", new ao::vulkan::CounterMetric<std::chrono::seconds, int>(0));
}

void FrustumDemo::render() {
    auto cpuFrame = static_cast<ao::vulkan::DurationMetric*>((*this->metrics)["CPU"]);
    auto fps = static_cast<ao::vulkan::CounterMetric<std::chrono::seconds, int>*>((*this->metrics)["Frame/s"]);
    auto triangle_count = static_cast<ao::vulkan::CounterCommandBufferMetric<std::chrono::seconds, u64>*>((*this->metrics)["Triangle/s"]);

    // Render
    cpuFrame->start();
    {
        vk::Fence fence = this->fences[this->current_frame];

        // Wait fence
        this->device->logical()->waitForFences(fence, VK_TRUE, (std::numeric_limits<u64>::max)());

        // Prepare frame
        this->prepareFrame();

        // Call	beforeCommandBuffersUpdate()
        this->beforeCommandBuffersUpdate();

        // Update command buffers
        this->updateCommandBuffers();

        /* COMPUTE PART */
        if (this->settings_->get<bool>(CullingEnableKey, true)) {
            auto sem_index = (ao::vulkan::semaphore::ComputeProcess * this->swapchain->size()) + this->current_frame;
            vk::SubmitInfo submit_info(static_cast<u32>(this->semaphores[sem_index].waits.size()),
                                       this->semaphores[sem_index].waits.empty() ? nullptr : this->semaphores[sem_index].waits.data(), nullptr, 3,
                                       &this->primary_compute_command_buffers[this->swapchain->frameIndex() * 3],
                                       static_cast<u32>(this->semaphores[sem_index].signals.size()),
                                       this->semaphores[sem_index].signals.empty() ? nullptr : this->semaphores[sem_index].signals.data());

            // Submit command buffers
            this->device->queues()->at(vk::to_string(vk::QueueFlagBits::eCompute)).value.submit(submit_info, nullptr);
        }

        /* GRAPHICS PART */
        {
            vk::PipelineStageFlags pipeline_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

            // Create submit info
            auto sem_index = ((this->settings_->get<bool>(CullingEnableKey, true) ? ao::vulkan::semaphore::GraphicProcessAfterCompute
                                                                                  : ao::vulkan::semaphore::GraphicProcess) *
                              this->swapchain->size()) +
                             this->current_frame;
            vk::SubmitInfo submit_info(static_cast<u32>(this->semaphores[sem_index].waits.size()),
                                       this->semaphores[sem_index].waits.empty() ? nullptr : this->semaphores[sem_index].waits.data(),
                                       &pipeline_stage, 1, &this->swapchain->currentCommand(),
                                       static_cast<u32>(this->semaphores[sem_index].signals.size()),
                                       this->semaphores[sem_index].signals.empty() ? nullptr : this->semaphores[sem_index].signals.data());

            // Reset fence
            this->device->logical()->resetFences(fence);

            // Submit command buffer
            this->device->queues()->at(vk::to_string(vk::QueueFlagBits::eGraphics)).value.submit(submit_info, fence);
        }

        // Submit frame
        this->submitFrame();

        // Increment frame index
        this->current_frame = (this->current_frame + 1) % this->swapchain->size();
    }
    cpuFrame->stop();
    fps->increment();
    triangle_count->update();
}

vk::RenderPass FrustumDemo::createRenderPass() {
    // Define attachments
    std::array<vk::AttachmentDescription, 2> attachments;
    attachments[0]
        .setFormat(this->swapchain->surfaceColorFormat())
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);
    attachments[1]
        .setFormat(ao::vulkan::utilities::bestDepthStencilFormat(this->device->physical()))
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    // Define references
    vk::AttachmentReference color_reference = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depth_reference = vk::AttachmentReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::SubpassDescription subpass = vk::SubpassDescription()
                                         .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                                         .setColorAttachmentCount(1)
                                         .setPColorAttachments(&color_reference)
                                         .setPDepthStencilAttachment(&depth_reference);
    vk::SubpassDependency dependency = vk::SubpassDependency()
                                           .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                                           .setDstSubpass(0)
                                           .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                                           .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                                           .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

    // Create render pass
    return this->device->logical()->createRenderPass(
        vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), static_cast<u32>(attachments.size()), attachments.data(), 1, &subpass, 1, &dependency));
}

void FrustumDemo::createPipelines() {
    {
        /* PIPELINE LAYOUT PART */

        // Create bindings
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings;
        bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);
        bindings[1] = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);

        // Create layout
        std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
        descriptor_set_layouts.push_back(this->device->logical()->createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<u32>(bindings.size()), bindings.data())));

        auto pipeline_layout = std::make_shared<ao::vulkan::PipelineLayout>(this->device->logical(), descriptor_set_layouts);

        /* PIPELINE PART */

        // Create shadermodules
        ao::vulkan::ShaderModule module(this->device->logical());

        // Load shaders & get shaderStages
        std::vector<vk::PipelineShaderStageCreateInfo> shader_stages =
            module.loadShader(vk::ShaderStageFlagBits::eVertex, "assets/shaders/frustum-culling/vert.spv")
                .loadShader(vk::ShaderStageFlagBits::eFragment, "assets/shaders/frustum-culling/frag.spv")
                .shaderStages();

        // Construct the different states making up the pipeline

        // Input assembly state
        vk::PipelineInputAssemblyStateCreateInfo input_state(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList);

        // Rasterization state
        vk::PipelineRasterizationStateCreateInfo rasterization_state = vk::PipelineRasterizationStateCreateInfo()
                                                                           .setPolygonMode(vk::PolygonMode::eFill)
                                                                           .setCullMode(vk::CullModeFlagBits::eBack)
                                                                           .setFrontFace(vk::FrontFace::eCounterClockwise)
                                                                           .setLineWidth(1.0f);

        // Color blend state
        std::array<vk::PipelineColorBlendAttachmentState, 1> blend_attachments;
        blend_attachments[0].setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                                               vk::ColorComponentFlagBits::eA);

        vk::PipelineColorBlendStateCreateInfo color_state = vk::PipelineColorBlendStateCreateInfo()
                                                                .setAttachmentCount(static_cast<u32>(blend_attachments.size()))
                                                                .setPAttachments(blend_attachments.data());

        // Viewport state
        vk::Viewport viewport(0, 0, static_cast<float>(this->swapchain->extent().width), static_cast<float>(this->swapchain->extent().height), 0, 1);
        vk::Rect2D scissor(vk::Offset2D(), this->swapchain->extent());
        vk::PipelineViewportStateCreateInfo viewport_state(vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissor);

        // Enable dynamic states
        std::vector<vk::DynamicState> dynamic_states;
        dynamic_states.push_back(vk::DynamicState::eViewport);
        dynamic_states.push_back(vk::DynamicState::eScissor);

        vk::PipelineDynamicStateCreateInfo dynamic_state(vk::PipelineDynamicStateCreateFlags(), static_cast<u32>(dynamic_states.size()),
                                                         dynamic_states.data());

        // Depth and stencil state
        vk::PipelineDepthStencilStateCreateInfo depth_stencil_state =
            vk::PipelineDepthStencilStateCreateInfo().setDepthTestEnable(VK_TRUE).setDepthWriteEnable(VK_TRUE).setDepthCompareOp(
                vk::CompareOp::eLess);

        // Multi sampling state
        vk::PipelineMultisampleStateCreateInfo multisample_state;

        // Vertex input descriptions
        // Specifies the vertex input parameters for a pipeline

        // Vertex input binding
        std::array<vk::VertexInputBindingDescription, 2> vertex_inputs;
        vertex_inputs[0] = vk::VertexInputBindingDescription(0, sizeof(TexturedVertex), vk::VertexInputRate::eVertex);
        vertex_inputs[1] = vk::VertexInputBindingDescription(1, sizeof(UBO::InstanceData), vk::VertexInputRate::eInstance);

        // Input attribute bindings
        std::array<vk::VertexInputAttributeDescription, 7> vertex_attributes;
        vertex_attributes[0] = vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(TexturedVertex, pos));
        vertex_attributes[1] = vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(TexturedVertex, texture_coord));

        // Matrice 4x4
        vertex_attributes[2] = vk::VertexInputAttributeDescription(2, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(UBO::InstanceData, rotation));
        vertex_attributes[3] =
            vk::VertexInputAttributeDescription(3, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(UBO::InstanceData, rotation) + sizeof(glm::vec4));
        vertex_attributes[4] =
            vk::VertexInputAttributeDescription(4, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(UBO::InstanceData, rotation) + sizeof(glm::vec4) * 2);
        vertex_attributes[5] =
            vk::VertexInputAttributeDescription(5, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(UBO::InstanceData, rotation) + sizeof(glm::vec4) * 3);

        vertex_attributes[6] =
            vk::VertexInputAttributeDescription(6, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(UBO::InstanceData, position_and_scale));

        // Vertex input state used for pipeline creation
        vk::PipelineVertexInputStateCreateInfo vertex_state(vk::PipelineVertexInputStateCreateFlags(), static_cast<u32>(vertex_inputs.size()),
                                                            vertex_inputs.data(), static_cast<u32>(vertex_attributes.size()),
                                                            vertex_attributes.data());

        // Cache create info
        auto cache = ao::vulkan::GLFWEngine::LoadCache("data/frustum-culling/caches/graphics-main.cache");
        vk::PipelineCacheCreateInfo cache_info(vk::PipelineCacheCreateFlags(), cache.size(), cache.data());

        // Create rendering pipeline using the specified states
        this->pipelines["graphics-main"] = new ao::vulkan::GraphicsPipeline(
            this->device->logical(), pipeline_layout, this->render_pass, shader_stages, vertex_state, input_state, std::nullopt, viewport_state,
            rasterization_state, multisample_state, depth_stencil_state, color_state, dynamic_state, cache_info);

        /* DESCRIPTOR POOL PART */

        std::array<vk::DescriptorPoolSize, 2> pool_sizes;
        pool_sizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, static_cast<u32>(this->swapchain->size()));
        pool_sizes[1] = vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, static_cast<u32>(this->swapchain->size()));

        this->pipelines["graphics-main"]->pools().push_back(std::move(ao::vulkan::DescriptorPool(
            this->device->logical(), vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size()),
                                                                  static_cast<u32>(pool_sizes.size()), pool_sizes.data()))));
    }

    {
        /* PIPELINE LAYOUT PART */

        // Create bindings
        std::array<vk::DescriptorSetLayoutBinding, 5> bindings;
        bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[1] = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[2] = vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[3] = vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[4] = vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);

        // Create layout
        std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
        descriptor_set_layouts.push_back(this->device->logical()->createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<u32>(bindings.size()), bindings.data())));

        auto pipeline_layout = std::make_shared<ao::vulkan::PipelineLayout>(this->device->logical(), descriptor_set_layouts);

        /* PIPELINE PART */

        // Create shadermodules
        ao::vulkan::ShaderModule module(this->device->logical());

        // Load shaders & get shaderStages
        std::vector<vk::PipelineShaderStageCreateInfo> shader_stages =
            module.loadShader(vk::ShaderStageFlagBits::eCompute, "assets/shaders/frustum-culling/comp.spv").shaderStages();

        // Cache create info
        auto cache = ao::vulkan::GLFWEngine::LoadCache("data/frustum-culling/caches/compute-frustum.cache");
        vk::PipelineCacheCreateInfo cache_info(vk::PipelineCacheCreateFlags(), cache.size(), cache.data());

        // Create rendering pipeline using the specified states
        this->pipelines["compute-frustum"] =
            new ao::vulkan::ComputePipeline(this->device->logical(), pipeline_layout, shader_stages.front(), cache_info);

        /* DESCRIPTOR POOL PART */

        std::array<vk::DescriptorPoolSize, 1> pool_sizes;
        pool_sizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, static_cast<u32>(this->swapchain->size()));

        this->pipelines["compute-frustum"]->pools().push_back(std::move(ao::vulkan::DescriptorPool(
            this->device->logical(), vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size()),
                                                                  static_cast<u32>(pool_sizes.size()), pool_sizes.data()))));
    }

    {
        /* PIPELINE LAYOUT PART */

        // Create bindings
        std::array<vk::DescriptorSetLayoutBinding, 4> bindings;
        bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[1] = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[2] = vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[3] = vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute);

        // Create layout
        std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
        descriptor_set_layouts.push_back(this->device->logical()->createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<u32>(bindings.size()), bindings.data())));

        auto pipeline_layout = std::make_shared<ao::vulkan::PipelineLayout>(this->device->logical(), descriptor_set_layouts);

        /* PIPELINE PART */

        // Create shadermodules
        ao::vulkan::ShaderModule module(this->device->logical());

        // Load shaders & get shaderStages
        std::vector<vk::PipelineShaderStageCreateInfo> shader_stages =
            module.loadShader(vk::ShaderStageFlagBits::eCompute, "assets/shaders/frustum-culling-setup/comp.spv").shaderStages();

        // Cache create info
        auto cache = ao::vulkan::GLFWEngine::LoadCache("data/frustum-culling/caches/compute-frustum-setup.cache");
        vk::PipelineCacheCreateInfo cache_info(vk::PipelineCacheCreateFlags(), cache.size(), cache.data());

        // Create rendering pipeline using the specified states
        this->pipelines["compute-frustum-setup"] =
            new ao::vulkan::ComputePipeline(this->device->logical(), pipeline_layout, shader_stages.front(), cache_info);

        /* DESCRIPTOR POOL PART */

        std::array<vk::DescriptorPoolSize, 2> pool_sizes;
        pool_sizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, static_cast<u32>(this->swapchain->size()));
        pool_sizes[1] = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, static_cast<u32>(this->swapchain->size()));

        this->pipelines["compute-frustum-setup"]->pools().push_back(std::move(ao::vulkan::DescriptorPool(
            this->device->logical(), vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size()),
                                                                  static_cast<u32>(pool_sizes.size()), pool_sizes.data()))));
    }

    {
        /* PIPELINE LAYOUT PART */

        // Create bindings
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings;
        bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[1] = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        bindings[2] = vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);

        // Create layout
        std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
        descriptor_set_layouts.push_back(this->device->logical()->createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<u32>(bindings.size()), bindings.data())));

        auto pipeline_layout = std::make_shared<ao::vulkan::PipelineLayout>(this->device->logical(), descriptor_set_layouts);

        /* PIPELINE PART */

        // Create shadermodules
        ao::vulkan::ShaderModule module(this->device->logical());

        // Load shaders & get shaderStages
        std::vector<vk::PipelineShaderStageCreateInfo> shader_stages =
            module.loadShader(vk::ShaderStageFlagBits::eCompute, "assets/shaders/frustum-culling-optimize/comp.spv").shaderStages();

        // Cache create info
        auto cache = ao::vulkan::GLFWEngine::LoadCache("data/frustum-culling/caches/frustum-culling-optimize.cache");
        vk::PipelineCacheCreateInfo cache_info(vk::PipelineCacheCreateFlags(), cache.size(), cache.data());

        // Create rendering pipeline using the specified states
        this->pipelines["frustum-culling-optimize"] =
            new ao::vulkan::ComputePipeline(this->device->logical(), pipeline_layout, shader_stages.front(), cache_info);

        /* DESCRIPTOR POOL PART */

        std::array<vk::DescriptorPoolSize, 1> pool_sizes;
        pool_sizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, static_cast<u32>(this->swapchain->size()));

        this->pipelines["frustum-culling-optimize"]->pools().push_back(std::move(ao::vulkan::DescriptorPool(
            this->device->logical(), vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size()),
                                                                  static_cast<u32>(pool_sizes.size()), pool_sizes.data()))));
    }

    // Define callback
    auto device = this->device;
    this->pipelines.setBeforePipelineCacheDestruction([this, device](std::string name, vk::PipelineCache cache) {
        this->saveCache("data/frustum-culling/caches", name + std::string(".cache"), cache);
    });
}

void FrustumDemo::createVulkanBuffers() {
    /* LOAD MODEL */

    // Load
    ObjFile model;
    this->LOGGER << ao::core::Logger::Level::trace << "=== Start loading model ===";
    if (!objParseFile(model, "assets/models/rock.obj")) {
        throw ao::core::Exception("Error during model loading");
    }
    this->indices_count = static_cast<u32>(model.f_size / 3);

    this->LOGGER << ao::core::Logger::Level::trace << fmt::format("Vertex count: {}", this->indices_count);

    // Prepare vector
    this->LOGGER << ao::core::Logger::Level::trace << "Filling vectors with model's data";
    std::vector<MeshOptVertex> opt_vertices(this->indices_count);

    // Build vertices vector
    for (size_t i = 0; i < this->indices_count; i++) {
        int vi = model.f[i * 3 + 0];
        int vti = model.f[i * 3 + 1];
        int vni = model.f[i * 3 + 2];

        opt_vertices[i] = {
            model.v[vi * 3 + 0],
            model.v[vi * 3 + 1],
            model.v[vi * 3 + 2],

            vni >= 0 ? model.vn[vni * 3 + 0] : 0,
            vni >= 0 ? model.vn[vni * 3 + 1] : 0,
            vni >= 0 ? model.vn[vni * 3 + 2] : 0,

            vti >= 0 ? model.vt[vti * 3 + 0] : 0,
            1.0f - (vti >= 0 ? model.vt[vti * 3 + 1] : 0),
        };
    }

    // Optimize mesh
    this->LOGGER << ao::core::Logger::Level::trace << "Optimize mesh";
    std::vector<u32> remap_indices(this->indices_count);
    size_t vertices_count = meshopt_generateVertexRemap(remap_indices.data(), nullptr, this->indices_count, opt_vertices.data(), this->indices_count,
                                                        sizeof(MeshOptVertex));

    this->indices.resize(this->indices_count);
    std::vector<MeshOptVertex> remap_vertices(vertices_count);

    meshopt_remapVertexBuffer(remap_vertices.data(), opt_vertices.data(), this->indices_count, sizeof(MeshOptVertex), remap_indices.data());
    meshopt_remapIndexBuffer(this->indices.data(), 0, this->indices_count, remap_indices.data());

    meshopt_optimizeVertexCache(this->indices.data(), this->indices.data(), this->indices_count, vertices_count);
    meshopt_optimizeVertexFetch(remap_vertices.data(), this->indices.data(), this->indices_count, remap_vertices.data(), vertices_count,
                                sizeof(MeshOptVertex));

    this->LOGGER << ao::core::Logger::Level::trace << fmt::format("Vertex count afer optimization: {}", vertices_count);

    // Convert into TexturedVertex
    this->LOGGER << ao::core::Logger::Level::trace << "Convert MeshOptVertex -> TexturedVertex";
    this->vertices.resize(vertices_count);
    for (size_t i = 0; i < vertices_count; i++) {
        vertices[i] = {{remap_vertices[i].px, remap_vertices[i].py, remap_vertices[i].pz}, {remap_vertices[i].tx, remap_vertices[i].ty}};
    }

    this->LOGGER << ao::core::Logger::Level::trace << "=== Model loading end ===";

    // Create vertices & indices
    this->model_buffer =
        std::make_unique<ao::vulkan::StagingTupleBuffer<TexturedVertex, u32>>(this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    this->model_buffer
        ->init({sizeof(TexturedVertex) * this->vertices.size(), sizeof(u32) * this->indices_count}, vk::BufferUsageFlagBits::eVertexBuffer)
        ->update(this->vertices.data(), this->indices.data());

    this->model_buffer->freeHostBuffer();

    // Free vectors
    this->vertices.resize(0);
    this->indices.resize(0);

    this->ubo_buffer = std::make_unique<ao::vulkan::BasicDynamicArrayBuffer<UBO>>(this->swapchain->size(), this->device);
    this->ubo_buffer->init(vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive, vk::MemoryPropertyFlagBits::eHostVisible,
                           ao::vulkan::Buffer::CalculateUBOAligmentSize(this->device->physical(), sizeof(UBO)));

    // Resize uniform buffers vector
    this->uniform_buffers.resize(this->swapchain->size());

    this->instance_buffer =
        std::make_unique<ao::vulkan::StagingTupleBuffer<UBO::InstanceData, float>>(this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    this->instance_buffer->init({INSTANCE_COUNT * sizeof(UBO::InstanceData), INSTANCE_COUNT * this->swapchain->size() * sizeof(float)},
                                vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer);

    auto draw_commands =
        std::vector<vk::DrawIndexedIndirectCommand>(this->swapchain->size(), vk::DrawIndexedIndirectCommand(this->indices_count, INSTANCE_COUNT));
    this->draw_command_buffer = std::make_unique<ao::vulkan::StagingDynamicArrayBuffer<vk::DrawIndexedIndirectCommand>>(
        this->swapchain->size(), this->device, vk::CommandBufferUsageFlagBits::eSimultaneousUse);
    this->draw_command_buffer->init(sizeof(vk::DrawIndexedIndirectCommand), vk::BufferUsageFlagBits::eStorageBuffer)->update(draw_commands);

    this->frustum_planes_buffer = std::make_unique<ao::vulkan::BasicDynamicArrayBuffer<float>>(this->swapchain->size() * 6, this->device);
    this->frustum_planes_buffer->init(vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive, vk::MemoryPropertyFlagBits::eDeviceLocal,
                                      sizeof(glm::vec4));

    auto dispatch_buffers = std::vector<vk::DispatchIndirectCommand>(this->swapchain->size(), vk::DispatchIndirectCommand(INSTANCE_COUNT, 1, 1));
    this->dispatch_buffer = std::make_unique<ao::vulkan::StagingDynamicArrayBuffer<vk::DispatchIndirectCommand>>(
        this->swapchain->size(), this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    this->dispatch_buffer->init(sizeof(vk::DispatchIndirectCommand), vk::BufferUsageFlagBits::eStorageBuffer)->update(dispatch_buffers);
    this->dispatch_buffer->freeHostBuffer();

    /* TEXTURE CREATION */

    // Load texture
    char* texture_file = "assets/textures/rock.png";
    int texture_width, texture_height, texture_channels;
    pixel_t* pixels = stbi_load(texture_file, &texture_width, &texture_height, &texture_channels, STBI_rgb_alpha);

    // Check image
    if (!pixels) {
        throw ao::core::Exception(fmt::format("Fail to load image: {0}", texture_file));
    }

    // Create buffer
    auto texture_buffer = ao::vulkan::BasicTupleBuffer<pixel_t>(this->device);
    texture_buffer
        .init(vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
              vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible,
              {texture_width * texture_height * 4 * sizeof(pixel_t)})
        ->update(pixels);

    // Free image
    stbi_image_free(pixels);

    // Create image
    auto image = ao::vulkan::utilities::createImage(
        *this->device->logical(), this->device->physical(), texture_width, texture_height, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageType::e2D,
        vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Assign
    std::get<0>(this->texture) = image.first;
    std::get<1>(this->texture) = image.second;

    // Process image & copy into image
    ao::vulkan::utilities::updateImageLayout(
        *this->device->logical(), this->device->graphicsPool(), *this->device->queues(), std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm,
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    ao::vulkan::utilities::copyBufferToImage(*this->device->logical(), this->device->transferPool(), *this->device->queues(), texture_buffer.buffer(),
                                             std::get<0>(this->texture),
                                             vk::BufferImageCopy(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                                                                 vk::Offset3D(), vk::Extent3D(vk::Extent2D(texture_width, texture_height), 1)));
    ao::vulkan::utilities::updateImageLayout(*this->device->logical(), this->device->graphicsPool(), *this->device->queues(),
                                             std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm,
                                             vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1),
                                             vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create view
    std::get<2>(this->texture) =
        ao::vulkan::utilities::createImageView(*this->device->logical(), std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm,
                                               vk::ImageViewType::e2D, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

    // Create sampler
    this->texture_sampler = this->device->logical()->createSampler(
        vk::SamplerCreateInfo(vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
                              vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0, VK_TRUE, 16,
                              VK_FALSE, vk::CompareOp::eAlways, 0, 0, vk::BorderColor::eFloatTransparentBlack, VK_FALSE));

    /* GRAPHICS DESCRIPTOR SETS CREATION */
    {
        // Create vector of layouts
        std::vector<vk::DescriptorSetLayout> layouts(this->swapchain->size(),
                                                     this->pipelines["graphics-main"]->layout()->descriptorLayouts().front());

        // Create sets
        auto descriptor_sets =
            this->pipelines["graphics-main"]->pools().front().allocateDescriptorSets(static_cast<u32>(this->swapchain->size()), layouts);

        // Configure
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            vk::DescriptorImageInfo sample_info(this->texture_sampler, std::get<2>(this->texture), vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::DescriptorBufferInfo buffer_info(this->ubo_buffer->buffer(), this->ubo_buffer->offset(i), sizeof(UBO));

            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info), {});

            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &sample_info), {});
        }
    }

    /* FRUSTUM COMPUTE DESCRIPTOR SETS CREATION */
    {
        // Create vector of layouts
        std::vector<vk::DescriptorSetLayout> layouts(this->swapchain->size(),
                                                     this->pipelines["compute-frustum"]->layout()->descriptorLayouts().front());

        // Create sets
        auto descriptor_sets =
            this->pipelines["compute-frustum"]->pools().front().allocateDescriptorSets(static_cast<u32>(this->swapchain->size()), layouts);

        // Configure
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            vk::DescriptorBufferInfo command_info(this->draw_command_buffer->buffer(), this->draw_command_buffer->offset(i),
                                                  sizeof(vk::DrawIndexedIndirectCommand));
            vk::DescriptorBufferInfo dispatch_info(this->dispatch_buffer->buffer(), this->dispatch_buffer->offset(i),
                                                   sizeof(vk::DispatchIndirectCommand));
            vk::DescriptorBufferInfo instance_enabled_info(this->instance_buffer->buffer(),
                                                           this->instance_buffer->offset(1) + (i * INSTANCE_COUNT * sizeof(float)),
                                                           INSTANCE_COUNT * sizeof(float));
            vk::DescriptorBufferInfo instances_info(this->instance_buffer->buffer(), 0, sizeof(UBO::InstanceData) * INSTANCE_COUNT);
            vk::DescriptorBufferInfo frustum_info(this->frustum_planes_buffer->buffer(), this->frustum_planes_buffer->offset(6 * i),
                                                  sizeof(glm::vec4) * 6);

            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &command_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dispatch_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &instance_enabled_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &instances_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &frustum_info), {});
        }
    }

    /* FRUSTUM SET-UP DESCRIPTOR SETS CREATION */
    {
        // Create vector of layouts
        std::vector<vk::DescriptorSetLayout> layouts(this->swapchain->size(),
                                                     this->pipelines["compute-frustum-setup"]->layout()->descriptorLayouts().front());

        // Create sets
        auto descriptor_sets =
            this->pipelines["compute-frustum-setup"]->pools().front().allocateDescriptorSets(static_cast<u32>(this->swapchain->size()), layouts);

        // Configure
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            vk::DescriptorBufferInfo command_info(this->draw_command_buffer->buffer(), this->draw_command_buffer->offset(i),
                                                  sizeof(vk::DrawIndexedIndirectCommand));
            vk::DescriptorBufferInfo dispatch_info(this->dispatch_buffer->buffer(), this->dispatch_buffer->offset(i),
                                                   sizeof(vk::DispatchIndirectCommand));
            vk::DescriptorBufferInfo frustum_info(this->frustum_planes_buffer->buffer(), this->frustum_planes_buffer->offset(6 * i),
                                                  sizeof(glm::vec4) * 6);
            vk::DescriptorBufferInfo ubo_info(this->ubo_buffer->buffer(), this->ubo_buffer->offset(i), sizeof(UBO));

            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &command_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dispatch_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &frustum_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 3, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &ubo_info), {});
        }
    }

    /* FRUSTUM OPTIMIZATION DESCRIPTOR SETS CREATION */
    {
        // Create vector of layouts
        std::vector<vk::DescriptorSetLayout> layouts(this->swapchain->size(),
                                                     this->pipelines["frustum-culling-optimize"]->layout()->descriptorLayouts().front());

        // Create sets
        auto descriptor_sets =
            this->pipelines["frustum-culling-optimize"]->pools().front().allocateDescriptorSets(static_cast<u32>(this->swapchain->size()), layouts);

        // Configure
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            vk::DescriptorBufferInfo command_info(this->draw_command_buffer->buffer(), this->draw_command_buffer->offset(i),
                                                  sizeof(vk::DrawIndexedIndirectCommand));
            vk::DescriptorBufferInfo instance_enabled_info(this->instance_buffer->buffer(),
                                                           this->instance_buffer->offset(1) + (i * INSTANCE_COUNT * sizeof(float)),
                                                           INSTANCE_COUNT * sizeof(float));
            vk::DescriptorBufferInfo instances_info(this->instance_buffer->buffer(), 0, sizeof(UBO::InstanceData) * INSTANCE_COUNT);

            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &command_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &instance_enabled_info), {});
            this->device->logical()->updateDescriptorSets(
                vk::WriteDescriptorSet(descriptor_sets[i], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &instances_info), {});
        }
    }
}

void FrustumDemo::createSecondaryCommandBuffers() {
    /* GRAPHICS SECONDARY */
    {
        auto command_buffers =
            this->secondary_command_pool->allocateCommandBuffers(vk::CommandBufferLevel::eSecondary, static_cast<u32>(this->swapchain->size()));

        this->secondary_command_buffers.resize(command_buffers.size());
        for (size_t i = 0; i < command_buffers.size(); i++) {
            this->secondary_command_buffers[i] = new ao::vulkan::GraphicsPrimaryCommandBuffer::SecondaryCommandBuffer(
                command_buffers[i],
                [pipeline = this->pipelines["graphics-main"], indices_count = this->indices_count, rectangles = this->model_buffer.get(),
                 instance = this->instance_buffer.get(), draw_indices = this->draw_command_buffer.get(),
                 swapchain_size = this->swapchain->size()](vk::CommandBuffer command_buffer, vk::CommandBufferInheritanceInfo const& inheritance_info,
                                                           vk::Extent2D swapchain_extent, int frame_index) {
                    // Begin info
                    vk::CommandBufferBeginInfo begin_info =
                        vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue).setPInheritanceInfo(&inheritance_info);

                    command_buffer.begin(begin_info);
                    {
                        // Set viewport & scissor
                        command_buffer.setViewport(
                            0, vk::Viewport(0, 0, static_cast<float>(swapchain_extent.width), static_cast<float>(swapchain_extent.height), 0, 1));
                        command_buffer.setScissor(0, vk::Rect2D(vk::Offset2D(), swapchain_extent));

                        // Bind pipeline
                        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->value());

                        // Draw rectangles
                        command_buffer.bindVertexBuffers(0, rectangles->buffer(), {0});
                        command_buffer.bindVertexBuffers(1, instance->buffer(), {0});
                        command_buffer.bindIndexBuffer(rectangles->buffer(), rectangles->offset(1), vk::IndexType::eUint32);
                        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout()->value(), 0,
                                                          pipeline->pools().front().descriptorSets().at(frame_index), {});

                        command_buffer.drawIndexedIndirect(draw_indices->buffer(), draw_indices->offset(frame_index), 1,
                                                           sizeof(vk::DrawIndexedIndirectCommand));
                    }
                    command_buffer.end();
                });
        }

        // Add to primary
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            this->primary_command_buffers[i]->addSecondary(this->secondary_command_buffers[i]);
        }
    }

    /* COMPUTE SECONDARY */
    {
        this->primary_compute_command_buffers =
            this->secondary_command_pool->allocateCommandBuffers(vk::CommandBufferLevel::ePrimary, static_cast<u32>(this->swapchain->size() * 3));

        this->compute_command_buffers.resize(this->primary_compute_command_buffers.size());
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            this->compute_command_buffers[i * 3] = new ComputeCommandBuffer(
                this->primary_compute_command_buffers[i * 3],
                [frame_index = i, pipeline = this->pipelines["compute-frustum-setup"], &metrics = this->metrics](vk::CommandBuffer command_buffer) {
                    vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

                    command_buffer.begin(begin_info);
                    {
                        // Reset pool
                        if (frame_index == 0) {
                            command_buffer.resetQueryPool(metrics->timestampQueryPool(), 2, 2);
                        }

                        // Statistics
                        if (frame_index == 0) {
                            command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, metrics->timestampQueryPool(), 2);
                        }

                        // Bind pipeline
                        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->value());
                        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->layout()->value(), 0,
                                                          pipeline->pools().front().descriptorSets().at(frame_index), {});

                        // Dispatch
                        command_buffer.dispatch(7, 1, 1);  // 1 = Reset stuff, 6 = calculate frustum planes
                    }
                    command_buffer.end();
                });
            this->compute_command_buffers[i * 3 + 1] = new ComputeCommandBuffer(
                this->primary_compute_command_buffers[i * 3 + 1],
                [& metrics = this->metrics, frame_index = i, pipeline = this->pipelines["compute-frustum"]](vk::CommandBuffer command_buffer) {
                    vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eRenderPassContinue);

                    command_buffer.begin(begin_info);
                    {
                        // Bind pipeline
                        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->value());
                        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->layout()->value(), 0,
                                                          pipeline->pools().front().descriptorSets().at(frame_index), {});

                        // Dispatch
                        command_buffer.dispatch(INSTANCE_COUNT, 1, 1);
                    }
                    command_buffer.end();
                });
            this->compute_command_buffers[i * 3 + 2] = new ComputeCommandBuffer(
                this->primary_compute_command_buffers[i * 3 + 2],
                [& metrics = this->metrics, frame_index = i, dispatch_buffer = this->dispatch_buffer.get(),
                 pipeline = this->pipelines["frustum-culling-optimize"]](vk::CommandBuffer command_buffer) {
                    vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

                    command_buffer.begin(begin_info);
                    {
                        // Bind pipeline
                        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->value());
                        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->layout()->value(), 0,
                                                          pipeline->pools().front().descriptorSets().at(frame_index), {});

                        // Dispatch
                        command_buffer.dispatchIndirect(dispatch_buffer->buffer(), dispatch_buffer->offset(frame_index));

                        // Statistics
                        if (frame_index == 0) {
                            command_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, metrics->timestampQueryPool(), 3);
                        }
                    }
                    command_buffer.end();
                });
        }

        // Update command buffers
        for (size_t i = 0; i < this->compute_command_buffers.size(); i++) {
            this->compute_command_buffers[i]->update();
        }
    }
}

void FrustumDemo::beforeCommandBuffersUpdate() {
    if (!this->clock_start) {
        this->clock = std::chrono::system_clock::now();
        this->clock_start = true;

        std::vector<UBO::InstanceData> instance_data(INSTANCE_COUNT);
        std::vector<float> instance_visible(this->swapchain->size() * INSTANCE_COUNT, 1.0f);

        // Init uniform buffers
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            this->uniform_buffers[i].proj =
                glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / (float)this->swapchain->extent().height, 0.1f, 100000.0f);
            this->uniform_buffers[i].proj[1][1] *= -1;  // Adapt for vulkan

            this->uniform_buffers[i].view = glm::lookAt(glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        }

        // Init instance data
        static constexpr float PositionInterval = .15f;
        static constexpr float RotationInterval = .5f;
        for (size_t i = 0; i < INSTANCE_COUNT; i++) {
            instance_data[i].rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, .0f, .0f));

            auto angle = (RotationInterval * i) + 8;
            instance_data[i].position_and_scale = glm::vec4((PositionInterval + PositionInterval * angle) * std::cos(angle),
                                                            (PositionInterval + PositionInterval * angle) * std::sin(angle), .0f, .25f);
        }

        // Update
        this->ubo_buffer->update(this->uniform_buffers);
        this->instance_buffer->update(instance_data.data(), instance_visible.data());

        this->instance_buffer->freeHostBuffer();

        return;
    }

    // Update uniform buffer
    if (this->swapchain->state() == ao::vulkan::SwapchainState::eReset) {
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            this->uniform_buffers[i].proj =
                glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / (float)this->swapchain->extent().height, 0.1f, 100000.0f);
            this->uniform_buffers[i].proj[1][1] *= -1;  // Adapt for vulkan
        }
        this->ubo_buffer->update(this->uniform_buffers);
    }
}