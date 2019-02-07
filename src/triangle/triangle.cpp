// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "triangle.h"

#include <ao/vulkan/engine/wrapper/pipeline/graphics_pipeline.h>

TriangleDemo::~TriangleDemo() {
    this->vertices_buffer.reset();
    this->indices_buffer.reset();
}

vk::RenderPass TriangleDemo::createRenderPass() {
    // Define attachments
    std::array<vk::AttachmentDescription, 1> attachments;
    attachments[0]
        .setFormat(this->swapchain->colorFormat())
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    // Define references
    vk::AttachmentReference colorReference = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpass = vk::SubpassDescription()
                                         .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                                         .setColorAttachmentCount(1)
                                         .setPColorAttachments(&colorReference);
    vk::SubpassDependency dependency = vk::SubpassDependency()
                                           .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                                           .setDstSubpass(0)
                                           .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                                           .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                                           .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

    // Create render pass
    return this->device->logical.createRenderPass(
        vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), static_cast<u32>(attachments.size()), attachments.data(), 1, &subpass, 1, &dependency));
}

void TriangleDemo::createPipelines() {
    // Create shadermodules
    ao::vulkan::ShaderModule module(this->device);

    // Load shaders & get shaderStages
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages =
        module.loadShader(vk::ShaderStageFlagBits::eVertex, "assets/shaders/triangle/vert.spv")
            .loadShader(vk::ShaderStageFlagBits::eFragment, "assets/shaders/triangle/frag.spv")
            .shaderStages();

    // Construct the differnent states making up the pipeline

    // Input assembly state
    vk::PipelineInputAssemblyStateCreateInfo input_state(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList);

    // Rasterization state
    vk::PipelineRasterizationStateCreateInfo rasterization_state = vk::PipelineRasterizationStateCreateInfo()
                                                                       .setPolygonMode(vk::PolygonMode::eFill)
                                                                       .setCullMode(vk::CullModeFlagBits::eNone)
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
    vk::PipelineDepthStencilStateCreateInfo depth_stencil_state;

    // Multi sampling state
    vk::PipelineMultisampleStateCreateInfo multisample_state;

    // Vertex input descriptions
    // Specifies the vertex input parameters for a pipeline

    // Vertex input binding
    vk::VertexInputBindingDescription vertex_binding = vk::VertexInputBindingDescription().setStride(sizeof(Vertex));

    // Inpute attribute bindings
    std::array<vk::VertexInputAttributeDescription, 2> vertex_attributes = Vertex::AttributeDescriptions();

    // Vertex input state used for pipeline creation
    vk::PipelineVertexInputStateCreateInfo vertex_state(vk::PipelineVertexInputStateCreateFlags(), 1, &vertex_binding,
                                                        static_cast<u32>(vertex_attributes.size()), vertex_attributes.data());

    // Create rendering pipeline using the specified states
    this->pipelines["main"] = new ao::vulkan::GraphicsPipeline(
        this->device, std::make_shared<ao::vulkan::PipelineLayout>(this->device), this->render_pass, shader_stages, vertex_state, input_state,
        std::nullopt, viewport_state, rasterization_state, multisample_state, depth_stencil_state, color_state, dynamic_state);
}

void TriangleDemo::createVulkanBuffers() {
    this->vertices_buffer = std::unique_ptr<ao::vulkan::TupleBuffer<Vertex>>(
        (new ao::vulkan::StagingTupleBuffer<Vertex>(this->device, vk::CommandBufferUsageFlagBits::eSimultaneousUse, true))
            ->init({sizeof(Vertex) * this->vertices.size()}, vk::BufferUsageFlags(vk::BufferUsageFlagBits::eVertexBuffer))
            ->update(this->vertices.data()));

    this->indices_buffer = std::unique_ptr<ao::vulkan::TupleBuffer<u16>>(
        (new ao::vulkan::StagingTupleBuffer<u16>(this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit))
            ->init({sizeof(u16) * this->indices.size()}, vk::BufferUsageFlags(vk::BufferUsageFlagBits::eIndexBuffer))
            ->update(this->indices.data()));
}

void TriangleDemo::createSecondaryCommandBuffers() {
    this->command_buffers =
        this->secondary_command_pool->allocateCommandBuffers(vk::CommandBufferLevel::eSecondary, static_cast<u32>(this->swapchain->size()));
}

void TriangleDemo::executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritanceInfo, int frameIndex, vk::CommandBuffer primaryCmd) {
    // Create info
    vk::CommandBufferBeginInfo beginInfo =
        vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue).setPInheritanceInfo(&inheritanceInfo);

    // Draw in command
    auto& commandBuffer = this->command_buffers[frameIndex];
    commandBuffer.begin(beginInfo);
    {
        // Set viewport & scissor
        commandBuffer.setViewport(
            0, vk::Viewport(0, 0, static_cast<float>(this->swapchain->extent().width), static_cast<float>(this->swapchain->extent().height), 0, 1));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(), this->swapchain->extent()));

        // Bind pipeline
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipelines["main"]->value());

        // Memory barrier
        vk::BufferMemoryBarrier barrier(vk::AccessFlags(), vk::AccessFlagBits::eVertexAttributeRead,
                                        this->device->queues[vk::to_string(vk::QueueFlagBits::eTransfer)].family_index,
                                        this->device->queues[vk::to_string(vk::QueueFlagBits::eGraphics)].family_index,
                                        this->vertices_buffer->buffer());
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eVertexInput, vk::DependencyFlags(), {},
                                      barrier, {});

        // Draw triangle
        commandBuffer.bindVertexBuffers(0, {this->vertices_buffer->buffer()}, {0});
        commandBuffer.bindIndexBuffer(this->indices_buffer->buffer(), 0, vk::IndexType::eUint16);

        commandBuffer.drawIndexed(static_cast<u32>(this->indices.size()), 1, 0, 0, 0);
    }
    commandBuffer.end();

    // Pass to primary
    primaryCmd.executeCommands(commandBuffer);
}

void TriangleDemo::beforeCommandBuffersUpdate() {
    if (!this->clock_start) {
        this->clock = std::chrono::system_clock::now();
        this->clock_start = true;

        return;
    }

    // Define rotate axis
    glm::vec3 zAxis(0.0f, 0.0f, 1.0f);

    // Delta time
    float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::system_clock::now() - this->clock).count();
    float angles = glm::half_pi<float>();  // Rotation in 1 second

    for (Vertex& vertice : this->vertices) {
        // To vec4
        glm::vec4 point(vertice.pos.x, vertice.pos.y, 0, 0);

        // Rotate point
        point = glm::rotate(angles * deltaTime, zAxis) * point;

        // Update vertice
        vertice.pos = glm::vec3(point.x, point.y, point.z);
    }

    // Update vertex buffer
    this->vertices_buffer->update(this->vertices.data());

    // Update clock
    this->clock = std::chrono::system_clock::now();
}
