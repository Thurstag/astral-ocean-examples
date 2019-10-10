// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "frustum_culling.h"

#include <ao/vulkan/pipeline/compute_pipeline.h>
#include <ao/vulkan/pipeline/graphics_pipeline.h>
#include <ao/vulkan/utilities/device.h>
#include <meshoptimizer.h>
#include <objparser.h>
#include <stb_image.h>
#include <boost/filesystem.hpp>
#include <boost/range/irange.hpp>

#include "../shared/metrics/counter_metric.hpp"
#include "../shared/metrics/duration_metric.hpp"
#include "../shared/metrics/lambda_metric.h"

static constexpr char const* CullingEnableKey = "culling.enable";

void FrustumDemo::createSemaphores() {
    this->semaphores = std::make_unique<ao::vulkan::SemaphoreContainer>(this->device->logical());

    // Create semaphores
    this->semaphores->resize(5 * this->swapchain->size());
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        vk::Semaphore acquire = this->device->logical()->createSemaphore(vk::SemaphoreCreateInfo());
        vk::Semaphore render = this->device->logical()->createSemaphore(vk::SemaphoreCreateInfo());
        vk::Semaphore compute = this->device->logical()->createSemaphore(vk::SemaphoreCreateInfo());

        // Fill container
        this->semaphores->at(ao::vulkan::semaphore::AcquireImage * this->swapchain->size() + i).signals.push_back(acquire);

        this->semaphores->at(ao::vulkan::semaphore::ComputeProcess * this->swapchain->size() + i).signals.push_back(compute);

        this->semaphores->at(ao::vulkan::semaphore::GraphicProcessAfterCompute * this->swapchain->size() + i).waits.push_back(acquire);
        this->semaphores->at(ao::vulkan::semaphore::GraphicProcessAfterCompute * this->swapchain->size() + i).waits.push_back(compute);
        this->semaphores->at(ao::vulkan::semaphore::GraphicProcessAfterCompute * this->swapchain->size() + i).signals.push_back(render);

        this->semaphores->at(ao::vulkan::semaphore::GraphicProcess * this->swapchain->size() + i).waits.push_back(acquire);
        this->semaphores->at(ao::vulkan::semaphore::GraphicProcess * this->swapchain->size() + i).signals.push_back(render);

        this->semaphores->at(ao::vulkan::semaphore::PresentImage * this->swapchain->size() + i).waits.push_back(render);
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

    // Create allocators
    this->createAllocators();

    // Create command pool
    this->secondary_command_pool = std::make_unique<ao::vulkan::CommandPool>(
        this->device->logical(), vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        this->device->queues()->at(vk::to_string(vk::QueueFlagBits::eGraphics)).family_index, ao::vulkan::CommandPoolAccessMode::eConcurrent);

    // Init metric module
    this->createMetrics();
    this->metrics->add("GPU(Compute)", new ao::vulkan::DurationCommandBufferMetric<std::milli, 2>(
                                           "ms", std::make_pair(this->device, this->metrics->timestampQueryPool())));

    // Schedule input listener
    this->scheduler.schedule(60, [&]() {
        auto state = glfwGetKey(window, GLFW_KEY_T);

        // Toggle culling
        if (this->key_last_state == GLFW_RELEASE && state == GLFW_PRESS) {
            this->settings_->get<bool>(CullingEnableKey) = !this->settings_->get<bool>(CullingEnableKey);

            // Update draw commands/dispatch buffers
            for (size_t i = 0; i < this->swapchain->size(); i++) {
                this->draw_command_buffer->at(i) = vk::DrawIndexedIndirectCommand(this->indices_count, INSTANCE_COUNT);
            }
            this->draw_command_buffer->invalidate(0, this->draw_command_buffer->size());

            // Toggle gpu(compute) metric
            if (this->settings_->get<bool>(CullingEnableKey)) {
                this->metrics->add("GPU(Compute)", new ao::vulkan::DurationCommandBufferMetric<std::milli, 2>(
                                                       "ms", std::make_pair(this->device, this->metrics->timestampQueryPool())));
            } else {
                this->metrics->remove("GPU(Compute)");
            }

            LOG_MSG(debug) << fmt::format("Frustum culling: {}", this->settings_->get<bool>(CullingEnableKey) ? "On" : "Off");
        }

        // Save state
        this->key_last_state = state;
    });
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
            vk::SubmitInfo submit_info(static_cast<u32>(this->semaphores->at(sem_index).waits.size()),
                                       this->semaphores->at(sem_index).waits.empty() ? nullptr : this->semaphores->at(sem_index).waits.data(),
                                       nullptr, 3, &this->primary_compute_command_buffers[this->swapchain->frameIndex() * 3],
                                       static_cast<u32>(this->semaphores->at(sem_index).signals.size()),
                                       this->semaphores->at(sem_index).signals.empty() ? nullptr : this->semaphores->at(sem_index).signals.data());

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
            vk::SubmitInfo submit_info(static_cast<u32>(this->semaphores->at(sem_index).waits.size()),
                                       this->semaphores->at(sem_index).waits.empty() ? nullptr : this->semaphores->at(sem_index).waits.data(),
                                       &pipeline_stage, 1, &this->swapchain->currentCommand(),
                                       static_cast<u32>(this->semaphores->at(sem_index).signals.size()),
                                       this->semaphores->at(sem_index).signals.empty() ? nullptr : this->semaphores->at(sem_index).signals.data());

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
    LOG_MSG(trace) << "=== Start loading model ===";
    if (!objParseFile(model, "assets/models/rock.obj")) {
        throw ao::core::Exception("Error during model loading");
    }
    this->indices_count = static_cast<u32>(model.f_size / 3);

    LOG_MSG(trace) << fmt::format("Vertex count: {}", this->indices_count);

    // Prepare vector
    LOG_MSG(trace) << "Filling vectors with model's data";
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
    LOG_MSG(trace) << "Optimize mesh";
    std::vector<u32> remap_indices(this->indices_count);
    this->vertices_count = meshopt_generateVertexRemap(remap_indices.data(), nullptr, this->indices_count, opt_vertices.data(), this->indices_count,
                                                       sizeof(MeshOptVertex));

    this->indices.resize(this->indices_count);
    std::vector<MeshOptVertex> remap_vertices(vertices_count);

    meshopt_remapVertexBuffer(remap_vertices.data(), opt_vertices.data(), this->indices_count, sizeof(MeshOptVertex), remap_indices.data());
    meshopt_remapIndexBuffer(this->indices.data(), 0, this->indices_count, remap_indices.data());

    meshopt_optimizeVertexCache(this->indices.data(), this->indices.data(), this->indices_count, vertices_count);
    meshopt_optimizeVertexFetch(remap_vertices.data(), this->indices.data(), this->indices_count, remap_vertices.data(), vertices_count,
                                sizeof(MeshOptVertex));

    LOG_MSG(trace) << fmt::format("Vertex count afer optimization: {}", vertices_count);

    // Convert into TexturedVertex
    LOG_MSG(trace) << "Convert MeshOptVertex -> TexturedVertex";
    this->vertices.resize(vertices_count);
    for (size_t i = 0; i < vertices_count; i++) {
        vertices[i] = {{remap_vertices[i].px, remap_vertices[i].py, remap_vertices[i].pz}, {remap_vertices[i].tx, remap_vertices[i].ty}};
    }

    LOG_MSG(trace) << "=== Model loading end ===";

    // Create vertices & indices
    this->model_buffer = std::make_unique<ao::vulkan::Vector<char>>(
        sizeof(TexturedVertex) * this->vertices.size() + sizeof(u32) * this->indices.size(), this->device_allocator,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer);

    std::copy(std::execution::par_unseq, this->vertices.data(), this->vertices.data() + this->vertices.size(),
              reinterpret_cast<TexturedVertex*>(&this->model_buffer->at(0)));
    std::copy(std::execution::par_unseq, this->indices.data(), this->indices.data() + this->indices.size(),
              reinterpret_cast<u32*>(&this->model_buffer->at(sizeof(TexturedVertex) * this->vertices.size())));
    this->model_buffer->invalidate(0, this->model_buffer->size());

    this->device_allocator->freeHost(this->model_buffer->info());

    // Free vectors
    this->vertices.resize(0);
    this->indices.resize(0);

    this->ubo_buffer =
        std::make_unique<ao::vulkan::Vector<UBO>>(this->swapchain->size(), this->host_uniform_allocator, vk::BufferUsageFlagBits::eUniformBuffer);

    this->instance_buffer = std::make_unique<ao::vulkan::Vector<char>>(
        INSTANCE_COUNT * sizeof(UBO::InstanceData) + INSTANCE_COUNT * this->swapchain->size() * sizeof(float), this->device_allocator,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer);

    this->draw_command_buffer = std::make_unique<ao::vulkan::Vector<vk::DrawIndexedIndirectCommand>>(
        this->swapchain->size(), vk::DrawIndexedIndirectCommand(this->indices_count, INSTANCE_COUNT), this->device_allocator,
        vk::BufferUsageFlagBits::eIndirectBuffer);

    this->frustum_planes_buffer =
        std::make_unique<ao::vulkan::Vector<glm::vec4>>(this->swapchain->size() * 6, this->device_allocator, vk::BufferUsageFlagBits::eStorageBuffer);
    this->device_allocator->freeHost(this->frustum_planes_buffer->info());  // TODO: DeviceOnlyAllocator

    this->dispatch_buffer = std::make_unique<ao::vulkan::Vector<vk::DispatchIndirectCommand>>(
        this->swapchain->size(), vk::DispatchIndirectCommand(INSTANCE_COUNT, 1, 1), this->device_allocator, vk::BufferUsageFlagBits::eIndirectBuffer);
    this->device_allocator->freeHost(this->dispatch_buffer->info());

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
    ao::vulkan::Vector<pixel_t> texture_buffer(texture_width * texture_height * 4, this->host_allocator, vk::BufferUsageFlagBits::eTransferSrc);

    auto range = boost::irange<u64>(0, texture_buffer.size());
    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [&](auto i) { texture_buffer[i] = pixels[i]; });
    texture_buffer.invalidate(0, texture_buffer.size());

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
    ao::vulkan::utilities::copyBufferToImage(*this->device->logical(), this->device->transferPool(), *this->device->queues(),
                                             texture_buffer.info().buffer, std::get<0>(this->texture),
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
            vk::DescriptorBufferInfo buffer_info(this->ubo_buffer->info().buffer, this->ubo_buffer->offset(i), sizeof(UBO));

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
            vk::DescriptorBufferInfo command_info(this->draw_command_buffer->info().buffer, this->draw_command_buffer->offset(i),
                                                  sizeof(vk::DrawIndexedIndirectCommand));
            vk::DescriptorBufferInfo dispatch_info(this->dispatch_buffer->info().buffer, this->dispatch_buffer->offset(i),
                                                   sizeof(vk::DispatchIndirectCommand));
            vk::DescriptorBufferInfo instance_enabled_info(
                this->instance_buffer->info().buffer,
                this->instance_buffer->offset(INSTANCE_COUNT * sizeof(UBO::InstanceData) + i * INSTANCE_COUNT * sizeof(float)),
                INSTANCE_COUNT * sizeof(float));
            vk::DescriptorBufferInfo instances_info(this->instance_buffer->info().buffer, this->instance_buffer->offset(0),
                                                    sizeof(UBO::InstanceData) * INSTANCE_COUNT);
            vk::DescriptorBufferInfo frustum_info(this->frustum_planes_buffer->info().buffer, this->frustum_planes_buffer->offset(6 * i),
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
            vk::DescriptorBufferInfo command_info(this->draw_command_buffer->info().buffer, this->draw_command_buffer->offset(i),
                                                  sizeof(vk::DrawIndexedIndirectCommand));
            vk::DescriptorBufferInfo dispatch_info(this->dispatch_buffer->info().buffer, this->dispatch_buffer->offset(i),
                                                   sizeof(vk::DispatchIndirectCommand));
            vk::DescriptorBufferInfo frustum_info(this->frustum_planes_buffer->info().buffer, this->frustum_planes_buffer->offset(6 * i),
                                                  sizeof(glm::vec4) * 6);
            vk::DescriptorBufferInfo ubo_info(this->ubo_buffer->info().buffer, this->ubo_buffer->offset(i), sizeof(UBO));

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
            vk::DescriptorBufferInfo command_info(this->draw_command_buffer->info().buffer, this->draw_command_buffer->offset(i),
                                                  sizeof(vk::DrawIndexedIndirectCommand));
            vk::DescriptorBufferInfo instance_enabled_info(
                this->instance_buffer->info().buffer,
                this->instance_buffer->offset(INSTANCE_COUNT * sizeof(UBO::InstanceData) + i * INSTANCE_COUNT * sizeof(float)),
                INSTANCE_COUNT * sizeof(float));
            vk::DescriptorBufferInfo instances_info(this->instance_buffer->info().buffer, this->instance_buffer->offset(0),
                                                    sizeof(UBO::InstanceData) * INSTANCE_COUNT);

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
                [pipeline = this->pipelines["graphics-main"], indices_count = this->indices_count, vertices_count = this->vertices_count,
                 rock = this->model_buffer.get(), instance = this->instance_buffer.get(), draw_indices = this->draw_command_buffer.get(),
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

                        // Draw rocks
                        command_buffer.bindVertexBuffers(0, rock->info().buffer, {0});
                        command_buffer.bindVertexBuffers(1, instance->info().buffer, {instance->offset(0)});
                        command_buffer.bindIndexBuffer(rock->info().buffer, rock->offset(sizeof(TexturedVertex) * vertices_count),
                                                       vk::IndexType::eUint32);
                        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout()->value(), 0,
                                                          pipeline->pools().front().descriptorSets().at(frame_index), {});

                        command_buffer.drawIndexedIndirect(draw_indices->info().buffer, draw_indices->offset(frame_index), 1,
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
                        command_buffer.dispatchIndirect(dispatch_buffer->info().buffer, dispatch_buffer->offset(frame_index));

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

        // Init uniform buffers
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            auto& ubo = this->ubo_buffer->at(i);

            ubo.view = glm::lookAt(glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.proj = glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / static_cast<float>(this->swapchain->extent().height),
                                        0.1f, 100000.0f);
            ubo.proj[1][1] *= -1;  // Adapt for vulkan
        }

        static constexpr float PositionInterval = .15f;
        static constexpr float RotationInterval = .5f;

        // Init instance data
        auto instance_data = reinterpret_cast<UBO::InstanceData*>(&this->instance_buffer->at(0));
        for (size_t i = 0; i < INSTANCE_COUNT; i++) {
            instance_data[i].rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, .0f, .0f));

            auto angle = (RotationInterval * i) + 8;
            instance_data[i].position_and_scale = glm::vec4((PositionInterval + PositionInterval * angle) * std::cos(angle),
                                                            (PositionInterval + PositionInterval * angle) * std::sin(angle), .0f, .25f);
        }

        auto instance_visible = reinterpret_cast<float*>(&this->instance_buffer->at(INSTANCE_COUNT * sizeof(UBO::InstanceData)));
        for (size_t i = 0; i < INSTANCE_COUNT * this->swapchain->size(); i++) {
            instance_visible[i] = 1.0f;
        }

        // Update
        this->ubo_buffer->invalidate(0, this->ubo_buffer->size());
        this->instance_buffer->invalidate(0, this->instance_buffer->size());

        this->device_allocator->freeHost(this->instance_buffer->info());

        return;
    }

    // Update uniform buffer
    if (this->swapchain->state() == ao::vulkan::SwapchainState::eReset) {
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            auto& ubo = this->ubo_buffer->at(i);

            ubo.proj = glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / static_cast<float>(this->swapchain->extent().height),
                                        0.1f, 100000.0f);
            ubo.proj[1][1] *= -1;  // Adapt for vulkan
        }
        this->ubo_buffer->invalidate(0, this->ubo_buffer->size());
    }
}