// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "n-rectangles.h"

#include <execution>

#include <ao/vulkan/pipeline/graphics_pipeline.h>
#include <ao/vulkan/utilities/device.h>
#include <boost/range/irange.hpp>

#include "../shared/metrics/counter_metric.hpp"

void RectanglesDemo::prepareVulkan() {
    ao::vulkan::Engine::prepareVulkan();

    // Create primary wrappers
    this->primary_command_buffers.resize(this->swapchain->size());
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        this->primary_command_buffers[i] = new ao::vulkan::GraphicsPrimaryCommandBuffer(
            this->swapchain->commandBuffers()[i], ao::vulkan::ExecutionPolicy::eParallelUnsequenced,
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

void RectanglesDemo::freeVulkan() {
    // Free buffers
    this->object_buffer.reset();
    this->ubo_buffer.reset();

    // Free wrappers
    for (auto buffer : this->secondary_command_buffers) {
        delete buffer;
    }

    ao::vulkan::GLFWEngine::freeVulkan();
}

vk::RenderPass RectanglesDemo::createRenderPass() {
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
    vk::AttachmentReference color_ref = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depth_ref = vk::AttachmentReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::SubpassDescription subpass = vk::SubpassDescription()
                                         .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                                         .setColorAttachmentCount(1)
                                         .setPColorAttachments(&color_ref)
                                         .setPDepthStencilAttachment(&depth_ref);
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

void RectanglesDemo::createPipelines() {
    /* PIPELINE LAYOUT PART */

    // Create bindings
    std::array<vk::DescriptorSetLayoutBinding, 1> bindings;
    bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);

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
        module.loadShader(vk::ShaderStageFlagBits::eVertex, "assets/shaders/rectangle/vert.spv")
            .loadShader(vk::ShaderStageFlagBits::eFragment, "assets/shaders/rectangle/frag.spv")
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
        vk::PipelineDepthStencilStateCreateInfo().setDepthTestEnable(VK_TRUE).setDepthWriteEnable(VK_TRUE).setDepthCompareOp(vk::CompareOp::eLess);

    // Multi sampling state
    vk::PipelineMultisampleStateCreateInfo multisample_state;

    // Vertex input descriptions
    // Specifies the vertex input parameters for a pipeline

    // Vertex input binding
    vk::VertexInputBindingDescription vertex_input = Vertex::BindingDescription();

    // Inpute attribute bindings
    std::array<vk::VertexInputAttributeDescription, 2> vertex_attributes = Vertex::AttributeDescriptions();
    // Vertex input state used for pipeline creation
    vk::PipelineVertexInputStateCreateInfo vertex_state(vk::PipelineVertexInputStateCreateFlags(), 1, &vertex_input,
                                                        static_cast<u32>(vertex_attributes.size()), vertex_attributes.data());

    // Cache create info
    auto cache = ao::vulkan::GLFWEngine::LoadCache("data/n-rectangles/caches/main.cache");
    vk::PipelineCacheCreateInfo cache_info(vk::PipelineCacheCreateFlags(), cache.size(), cache.data());

    // Create rendering pipeline using the specified states
    this->pipelines["main"] = new ao::vulkan::GraphicsPipeline(this->device->logical(), pipeline_layout, this->render_pass, shader_stages,
                                                               vertex_state, input_state, std::nullopt, viewport_state, rasterization_state,
                                                               multisample_state, depth_stencil_state, color_state, dynamic_state, cache_info);

    // Define callback
    auto device = this->device;
    this->pipelines.setBeforePipelineCacheDestruction([this, device](std::string name, vk::PipelineCache cache) {
        this->saveCache("data/n-rectangles/caches", name + std::string(".cache"), cache);
    });

    /* DESCRIPTOR POOL PART */

    std::array<vk::DescriptorPoolSize, 1> pool_sizes;
    pool_sizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, static_cast<u32>(this->swapchain->size()));

    this->pipelines["main"]->pools().push_back(ao::vulkan::DescriptorPool(
        this->device->logical(),
        vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size() * RECTANGLE_COUNT),
                                     static_cast<u32>(pool_sizes.size()), pool_sizes.data())));
}

void RectanglesDemo::createVulkanBuffers() {
    // Create vertices & indices
    this->object_buffer = std::make_unique<ao::vulkan::Vector<char>>(sizeof(Vertex) * this->vertices.size() + sizeof(u16) * this->indices.size(),
                                                                     this->device_allocator,
                                                                     vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer);
    std::copy(this->vertices.data(), this->vertices.data() + this->vertices.size(), reinterpret_cast<Vertex*>(&this->object_buffer->at(0)));

    std::copy(this->indices.data(), this->indices.data() + this->indices.size(),
              reinterpret_cast<u16*>(&this->object_buffer->at(sizeof(Vertex) * this->vertices.size())));
    this->object_buffer->invalidate(0, this->object_buffer->size());

    // Free host buffer
    this->device_allocator->freeHost(this->object_buffer->info());

    this->ubo_buffer = std::make_unique<ao::vulkan::Vector<UniformBufferObject>>(
        this->swapchain->size() * RECTANGLE_COUNT, this->host_uniform_allocator, vk::BufferUsageFlagBits::eUniformBuffer);

    /* DESCRIPTOR SETS CREATION */

    // Create vector of layouts
    std::vector<vk::DescriptorSetLayout> layouts(this->swapchain->size() * RECTANGLE_COUNT,
                                                 this->pipelines["main"]->layout()->descriptorLayouts().front());

    // Create sets
    auto descriptor_sets =
        this->pipelines["main"]->pools().front().allocateDescriptorSets(static_cast<u32>(this->swapchain->size() * RECTANGLE_COUNT), layouts);

    // Configure
    for (size_t i = 0; i < this->swapchain->size() * RECTANGLE_COUNT; i++) {
        vk::DescriptorBufferInfo buffer_info(this->ubo_buffer->info().buffer, this->ubo_buffer->offset(i), sizeof(UniformBufferObject));

        this->device->logical()->updateDescriptorSets(
            vk::WriteDescriptorSet(descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info), {});
    }
}

void RectanglesDemo::createSecondaryCommandBuffers() {
    auto command_buffers = this->secondary_command_pool->allocateCommandBuffers(vk::CommandBufferLevel::eSecondary,
                                                                                static_cast<u32>(RECTANGLE_COUNT * this->swapchain->size()));

    this->secondary_command_buffers.resize(command_buffers.size());
    for (size_t i = 0; i < command_buffers.size(); i++) {
        this->secondary_command_buffers[i] = new ao::vulkan::GraphicsPrimaryCommandBuffer::SecondaryCommandBuffer(
            command_buffers[i], [pipeline = this->pipelines["main"], indices_count = this->indices.size(), vertices_count = this->vertices.size(),
                                 object = this->object_buffer.get(), &ubo_buffer = this->ubo_buffer,
                                 cmd_index = i](vk::CommandBuffer command_buffer, vk::CommandBufferInheritanceInfo const& inheritance_info,
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
                    command_buffer.bindVertexBuffers(0, object->info().buffer, {0});
                    command_buffer.bindIndexBuffer(object->info().buffer, object->offset(sizeof(Vertex) * vertices_count), vk::IndexType::eUint16);
                    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout()->value(), 0,
                                                      pipeline->pools().front().descriptorSets()[cmd_index], {});

                    command_buffer.drawIndexed(static_cast<u32>(indices_count), 1, 0, 0, 0);
                }
                command_buffer.end();
            });
    }

    // Add to primary
    for (size_t i = 0; i < RECTANGLE_COUNT; i++) {
        for (size_t j = 0; j < this->swapchain->size(); j++) {
            this->primary_command_buffers[j]->addSecondary(this->secondary_command_buffers[(i * this->swapchain->size()) + j]);
        }
    }
}

void RectanglesDemo::beforeCommandBuffersUpdate() {
    if (!this->clock_start) {
        this->clock = std::chrono::system_clock::now();
        this->clock_start = true;

        // Generate vector of rotations & init uniform buffers
        std::srand(std::time(nullptr));
        this->rotations.resize(RECTANGLE_COUNT);
        for (size_t i = 0; i < RECTANGLE_COUNT; i++) {
            this->rotations[i] = std::make_pair(static_cast<float>((std::rand() % (180 - 10)) + 10), glm::vec3(0.f, 1.0f, 1.0f));

            for (size_t j = 0; j < this->swapchain->size(); j++) {
                auto& ubo = this->ubo_buffer->at((i * this->swapchain->size()) + j);

                ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
                ubo.proj = glm::perspective(glm::radians(45.0f),
                                            this->swapchain->extent().width / static_cast<float>(this->swapchain->extent().height), 0.1f, 10.0f);
                ubo.proj[1][1] *= -1;  // Adapt for vulkan
            }
        }
        this->ubo_buffer->invalidate(0, this->ubo_buffer->size());
        return;
    }

    // Delta time
    float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::system_clock::now() - this->clock).count();

    if (this->swapchain->state() == ao::vulkan::SwapchainState::eReset) {
        auto range = boost::irange<u64>(0, this->ubo_buffer->size());
        std::for_each(std::execution::par_unseq, range.begin(), range.end(), [&](auto i) {
            auto& ubo = this->ubo_buffer->at(i);

            ubo.proj = glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / static_cast<float>(this->swapchain->extent().height),
                                        0.1f, 10.0f);
            ubo.proj[1][1] *= -1;  // Adapt for vulkan
        });
    }

    // Update uniform buffers
    auto range = boost::irange<u64>(0, RECTANGLE_COUNT);
    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [&](auto i) {
        auto& ubo = this->ubo_buffer->at((i * this->swapchain->size()) + this->swapchain->frameIndex());

        ubo.rotation = glm::rotate(glm::mat4(1.0f), delta_time * glm::radians(this->rotations[i].first), this->rotations[i].second);
        ubo.scale = (0.25f * glm::cos(delta_time)) + 0.75f;
    });
    this->ubo_buffer->invalidate(0, this->ubo_buffer->size());
}
