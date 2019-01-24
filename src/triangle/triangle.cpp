// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "triangle.h"

TriangleDemo::~TriangleDemo() {
    this->vertexBuffer.reset();
    this->indexBuffer.reset();
}

void TriangleDemo::render() {
    ao::vulkan::GLFWEngine::render();

    if (!this->clockInit) {
        this->clock = std::chrono::system_clock::now();
        this->clockInit = true;

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
        vertice.pos = glm::vec2(point.x, point.y);
    }

    // Update vertex buffer
    this->vertexBuffer->update(this->vertices.data());

    // Update clock
    this->clock = std::chrono::system_clock::now();
}

void TriangleDemo::setUpRenderPass() {
    std::array<vk::AttachmentDescription, 2> attachments;

    // Color attachment
    attachments[0] = vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), this->swapchain->color_format, vk::SampleCountFlagBits::e1,
                                               vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
                                               vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

    // Depth attachment
    attachments[1] =
        vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), this->device->depth_format, vk::SampleCountFlagBits::e1,
                                  vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eClear,
                                  vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::AttachmentReference colorReference(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depthReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::SubpassDescription subpassDescription(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorReference,
                                              nullptr, &depthReference);

    // Subpass dependencies for layout transitions
    std::array<vk::SubpassDependency, 2> dependencies;

    dependencies[0] =
        vk::SubpassDependency(VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                              vk::DependencyFlagBits::eByRegion);

    dependencies[1] =
        vk::SubpassDependency(0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                              vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eMemoryRead,
                              vk::DependencyFlagBits::eByRegion);

    this->renderPass = this->device->logical.createRenderPass(
        vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), static_cast<u32>(attachments.size()), attachments.data(), 1, &subpassDescription,
                                 static_cast<u32>(dependencies.size()), dependencies.data()));
}

void TriangleDemo::createPipelineLayouts() {
    this->pipeline->layouts.resize(1);
    this->pipeline->layouts[0] = this->device->logical.createPipelineLayout(vk::PipelineLayoutCreateInfo());
}

void TriangleDemo::setUpPipelines() {
    // Create shadermodules
    ao::vulkan::ShaderModule module(this->device);

    // Load shaders & get shaderStages
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = module.loadShader("data/tri-vert.spv", vk::ShaderStageFlagBits::eVertex)
                                                                      .loadShader("data/tri-frag.spv", vk::ShaderStageFlagBits::eFragment)
                                                                      .shaderStages();

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
        vk::GraphicsPipelineCreateInfo().setLayout(this->pipeline->layouts[0]).setRenderPass(this->renderPass);

    // Construct the differnent states making up the pipeline

    // Set pipeline shader stage info
    pipelineCreateInfo.stageCount = static_cast<u32>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();

    // Input assembly state
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList);

    // Rasterization state
    vk::PipelineRasterizationStateCreateInfo rasterizationState = vk::PipelineRasterizationStateCreateInfo()
                                                                      .setPolygonMode(vk::PolygonMode::eFill)
                                                                      .setCullMode(vk::CullModeFlagBits::eNone)
                                                                      .setFrontFace(vk::FrontFace::eCounterClockwise)
                                                                      .setLineWidth(1.0f);

    // Color blend state
    std::array<vk::PipelineColorBlendAttachmentState, 1> blendAttachmentState;
    blendAttachmentState[0].setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                                              vk::ColorComponentFlagBits::eA);

    vk::PipelineColorBlendStateCreateInfo colorBlendState = vk::PipelineColorBlendStateCreateInfo()
                                                                .setAttachmentCount(static_cast<u32>(blendAttachmentState.size()))
                                                                .setPAttachments(blendAttachmentState.data());

    // Viewport state
    vk::Viewport viewport(0, 0, static_cast<float>(this->swapchain->current_extent.width), static_cast<float>(this->swapchain->current_extent.height),
                          0, 1);
    vk::Rect2D scissor(vk::Offset2D(), this->swapchain->current_extent);
    vk::PipelineViewportStateCreateInfo viewportState(vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissor);

    // Enable dynamic states
    std::vector<vk::DynamicState> dynamicStateEnables;
    dynamicStateEnables.push_back(vk::DynamicState::eViewport);
    dynamicStateEnables.push_back(vk::DynamicState::eScissor);

    vk::PipelineDynamicStateCreateInfo dynamicState(vk::PipelineDynamicStateCreateFlags(), static_cast<u32>(dynamicStateEnables.size()),
                                                    dynamicStateEnables.data());

    // Depth and stencil state
    vk::PipelineDepthStencilStateCreateInfo depthStencilState =
        vk::PipelineDepthStencilStateCreateInfo()
            .setDepthTestEnable(VK_TRUE)
            .setDepthWriteEnable(VK_TRUE)
            .setDepthCompareOp(vk::CompareOp::eLessOrEqual)
            .setBack(vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways));
    depthStencilState.setFront(depthStencilState.back);

    // Multi sampling state
    vk::PipelineMultisampleStateCreateInfo multisampleState;

    // Vertex input descriptions
    // Specifies the vertex input parameters for a pipeline

    // Vertex input binding
    vk::VertexInputBindingDescription vertexInputBinding = vk::VertexInputBindingDescription().setStride(sizeof(Vertex));

    // Inpute attribute bindings
    std::array<vk::VertexInputAttributeDescription, 2> vertexInputAttributes;

    vertexInputAttributes[0].setFormat(vk::Format::eR32G32Sfloat).setOffset(offsetof(Vertex, pos));

    vertexInputAttributes[1].setLocation(1).setFormat(vk::Format::eR32G32B32Sfloat).setOffset(offsetof(Vertex, color));

    // Vertex input state used for pipeline creation
    vk::PipelineVertexInputStateCreateInfo vertexInputState(vk::PipelineVertexInputStateCreateFlags(), 1, &vertexInputBinding,
                                                            static_cast<u32>(vertexInputAttributes.size()), vertexInputAttributes.data());

    // Assign the pipeline states to the pipeline creation info structure
    pipelineCreateInfo.setPVertexInputState(&vertexInputState)
        .setPInputAssemblyState(&inputAssemblyState)
        .setPRasterizationState(&rasterizationState)
        .setPColorBlendState(&colorBlendState)
        .setPMultisampleState(&multisampleState)
        .setPViewportState(&viewportState)
        .setPDepthStencilState(&depthStencilState)
        .setRenderPass(this->renderPass)
        .setPDynamicState(&dynamicState);

    // Create rendering pipeline using the specified states
    this->pipeline->pipelines = this->device->logical.createGraphicsPipelines(this->pipeline->cache, pipelineCreateInfo);
}

void TriangleDemo::setUpVulkanBuffers() {
    this->vertexBuffer = std::unique_ptr<ao::vulkan::TupleBuffer<Vertex>>(
        (new ao::vulkan::StagingTupleBuffer<Vertex>(this->device, vk::CommandBufferUsageFlagBits::eSimultaneousUse, true))
            ->init({sizeof(Vertex) * this->vertices.size()}, vk::BufferUsageFlags(vk::BufferUsageFlagBits::eVertexBuffer))
            ->update(this->vertices.data()));

    this->indexBuffer = std::unique_ptr<ao::vulkan::TupleBuffer<u16>>(
        (new ao::vulkan::StagingTupleBuffer<u16>(this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit))
            ->init({sizeof(u16) * this->indices.size()}, vk::BufferUsageFlags(vk::BufferUsageFlagBits::eIndexBuffer))
            ->update(this->indices.data()));
}

void TriangleDemo::createSecondaryCommandBuffers() {
    // Allocate buffers
    std::vector<vk::CommandBuffer> buffers = this->device->logical.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(this->swapchain->command_pool, vk::CommandBufferLevel::eSecondary, 1));

    // Add to container
    this->swapchain->commands["secondary"] = ao::vulkan::structs::CommandData(buffers, this->swapchain->command_pool);
}

std::vector<ao::vulkan::DrawInCommandBuffer> TriangleDemo::updateSecondaryCommandBuffers() {
    std::vector<ao::vulkan::DrawInCommandBuffer> commands;

    commands.push_back([this](int frameIndex, vk::CommandBufferInheritanceInfo const& inheritance,
                              std::pair<std::array<vk::ClearValue, 2>, vk::Rect2D> const& helpers) {
        vk::CommandBufferBeginInfo beginInfo =
            vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue).setPInheritanceInfo(&inheritance);
        vk::Viewport viewPort(0, 0, static_cast<float>(helpers.second.extent.width), static_cast<float>(helpers.second.extent.height), 0, 1);

        vk::CommandBuffer& commandBuffer = this->swapchain->commands["secondary"].buffers[0];

        // Draw in command
        commandBuffer.begin(beginInfo);
        {
            // Set viewport & scissor
            commandBuffer.setViewport(0, viewPort);
            commandBuffer.setScissor(0, helpers.second);

            // Bind pipeline
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipeline->pipelines[0]);

            // Memory barrier
            vk::BufferMemoryBarrier barrier(vk::AccessFlags(), vk::AccessFlagBits::eVertexAttributeRead,
                                            this->device->queues[vk::QueueFlagBits::eTransfer].index,
                                            this->device->queues[vk::QueueFlagBits::eGraphics].index, this->vertexBuffer->buffer());
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eVertexInput, vk::DependencyFlags(), {},
                                          barrier, {});

            // Draw triangle
            commandBuffer.bindVertexBuffers(0, {this->vertexBuffer->buffer()}, {0});
            commandBuffer.bindIndexBuffer(this->indexBuffer->buffer(), 0, vk::IndexType::eUint16);

            commandBuffer.drawIndexed(static_cast<u32>(this->indices.size()), 1, 0, 0, 0);
        }
        commandBuffer.end();

        return commandBuffer;
    });

    return commands;
}

void TriangleDemo::updateUniformBuffers() {}

vk::QueueFlags TriangleDemo::queueFlags() const {
    return ao::vulkan::GLFWEngine::queueFlags() | vk::QueueFlagBits::eTransfer;  // Enable transfer
}

void TriangleDemo::createDescriptorSetLayouts() {}

void TriangleDemo::createDescriptorPools() {}

void TriangleDemo::createDescriptorSets() {}
