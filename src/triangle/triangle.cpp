// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "triangle.h"

TriangleDemo::~TriangleDemo() {
    this->vertexBuffer.reset();
    this->indexBuffer.reset();
}

void TriangleDemo::setUpRenderPass() {
    // Define attachments
    std::array<vk::AttachmentDescription, 1> attachments;
    attachments[0]
        .setFormat(this->swapchain->color_format)
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
    this->renderPass = this->device->logical.createRenderPass(
        vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), static_cast<u32>(attachments.size()), attachments.data(), 1, &subpass, 1, &dependency));
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
    vk::PipelineDepthStencilStateCreateInfo depthStencilState;

    // Multi sampling state
    vk::PipelineMultisampleStateCreateInfo multisampleState;

    // Vertex input descriptions
    // Specifies the vertex input parameters for a pipeline

    // Vertex input binding
    vk::VertexInputBindingDescription vertexInputBinding = vk::VertexInputBindingDescription().setStride(sizeof(Vertex));

    // Inpute attribute bindings
    std::array<vk::VertexInputAttributeDescription, 2> vertexInputAttributes = Vertex::AttributeDescriptions();

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

void TriangleDemo::createVulkanBuffers() {
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

void TriangleDemo::executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritanceInfo, int frameIndex, vk::CommandBuffer primaryCmd) {
    // Create info
    vk::CommandBufferBeginInfo beginInfo =
        vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue).setPInheritanceInfo(&inheritanceInfo);

    // Draw in command
    auto& commandBuffer = this->swapchain->commands["secondary"].buffers[0];
    commandBuffer.begin(beginInfo);
    {
        // Set viewport & scissor
        commandBuffer.setViewport(0, vk::Viewport(0, 0, static_cast<float>(this->swapchain->current_extent.width),
                                                  static_cast<float>(this->swapchain->current_extent.height), 0, 1));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(), this->swapchain->current_extent));

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

    // Pass to primary
    primaryCmd.executeCommands(commandBuffer);
}

void TriangleDemo::beforeCommandBuffersUpdate() {
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
        vertice.pos = glm::vec3(point.x, point.y, point.z);
    }

    // Update vertex buffer
    this->vertexBuffer->update(this->vertices.data());

    // Update clock
    this->clock = std::chrono::system_clock::now();
}

vk::QueueFlags TriangleDemo::queueFlags() const {
    return ao::vulkan::GLFWEngine::queueFlags() | vk::QueueFlagBits::eTransfer;  // Enable transfer
}

void TriangleDemo::createDescriptorSetLayouts() {}

void TriangleDemo::createDescriptorPools() {}

void TriangleDemo::createDescriptorSets() {}