// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "n-rectangle.h"

#define RECTANGLE_COUNT 10
static_assert(RECTANGLE_COUNT <= 10);  // if RECTANGLE_COUNT exceeds 10 it will exceed descriptor layouts max count

RectangleDemo::~RectangleDemo() {
    this->rectangleBuffer.reset();
    this->uniformBuffer.reset();
}

void RectangleDemo::setUpRenderPass() {
    // Define attachments
    std::array<vk::AttachmentDescription, 2> attachments;
    attachments[0]
        .setFormat(this->swapchain->color_format)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);
    attachments[1]
        .setFormat(this->device->depth_format)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    // Define references
    vk::AttachmentReference colorReference = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depthReference = vk::AttachmentReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::SubpassDescription subpass = vk::SubpassDescription()
                                         .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                                         .setColorAttachmentCount(1)
                                         .setPColorAttachments(&colorReference)
                                         .setPDepthStencilAttachment(&depthReference);
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

void RectangleDemo::createPipelineLayouts() {
    this->pipeline->layouts.resize(1);

    this->pipeline->layouts.front() = this->device->logical.createPipelineLayout(vk::PipelineLayoutCreateInfo(
        vk::PipelineLayoutCreateFlags(), static_cast<u32>(this->descriptorSetLayouts.size()), this->descriptorSetLayouts.data()));
}

void RectangleDemo::setUpPipelines() {
    // Create shadermodules
    ao::vulkan::ShaderModule module(this->device);

    // Load shaders & get shaderStages
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = module.loadShader("data/rec-vert.spv", vk::ShaderStageFlagBits::eVertex)
                                                                      .loadShader("data/rec-frag.spv", vk::ShaderStageFlagBits::eFragment)
                                                                      .shaderStages();

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
        vk::GraphicsPipelineCreateInfo().setLayout(this->pipeline->layouts[0]).setRenderPass(this->renderPass);

    // Construct the different states making up the pipeline

    // Set pipeline shader stage info
    pipelineCreateInfo.stageCount = static_cast<u32>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();

    // Input assembly state
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList);

    // Rasterization state
    vk::PipelineRasterizationStateCreateInfo rasterizationState = vk::PipelineRasterizationStateCreateInfo()
                                                                      .setPolygonMode(vk::PolygonMode::eFill)
                                                                      .setCullMode(vk::CullModeFlagBits::eBack)
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
        .setRenderPass(renderPass)
        .setPDynamicState(&dynamicState);

    // Create rendering pipeline using the specified states
    this->pipeline->pipelines = this->device->logical.createGraphicsPipelines(this->pipeline->cache, pipelineCreateInfo);
}

void RectangleDemo::createVulkanBuffers() {
    // Create vertices & indices
    this->rectangleBuffer = std::unique_ptr<ao::vulkan::TupleBuffer<Vertex, u16>>(
        (new ao::vulkan::StagingTupleBuffer<Vertex, u16>(this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit))
            ->init({sizeof(Vertex) * this->vertices.size(), sizeof(u16) * this->indices.size()})
            ->update(this->vertices.data(), this->indices.data()));

    this->uniformBuffer = std::unique_ptr<ao::vulkan::DynamicArrayBuffer<UniformBufferObject>>(
        (new ao::vulkan::BasicDynamicArrayBuffer<UniformBufferObject>(this->swapchain->buffers.size() * RECTANGLE_COUNT, this->device))
            ->init(vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive, vk::MemoryPropertyFlagBits::eHostVisible,
                   ao::vulkan::Buffer::CalculateUBOAligmentSize(this->device->physical, sizeof(UniformBufferObject))));

    // Map buffer
    this->uniformBuffer->map();

    // Resize uniform buffers vector
    this->_uniformBuffers.resize(this->swapchain->buffers.size() * RECTANGLE_COUNT);
}

void RectangleDemo::createSecondaryCommandBuffers() {
    // Allocate buffers
    std::vector<vk::CommandBuffer> buffers = this->device->logical.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(this->swapchain->command_pool, vk::CommandBufferLevel::eSecondary, RECTANGLE_COUNT));

    // Add to container
    this->swapchain->commands["secondary"] = ao::vulkan::structs::CommandData(buffers, this->swapchain->command_pool);
}

void RectangleDemo::executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritanceInfo, int frameIndex, vk::CommandBuffer primaryCmd) {
    // Create info
    vk::CommandBufferBeginInfo beginInfo =
        vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue).setPInheritanceInfo(&inheritanceInfo);
    ao::vulkan::TupleBuffer<Vertex, u16>* rectangle = this->rectangleBuffer.get();

    // Draw in commands (TODO: Parallel mode)
    for (size_t i = 0; i < RECTANGLE_COUNT; i++) {
        auto& commandBuffer = this->swapchain->commands["secondary"].buffers[i];

        commandBuffer.begin(beginInfo);
        {
            // Set viewport & scissor
            commandBuffer.setViewport(0, vk::Viewport(0, 0, static_cast<float>(this->swapchain->current_extent.width),
                                                      static_cast<float>(this->swapchain->current_extent.height), 0, 1));
            commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(), this->swapchain->current_extent));

            // Bind pipeline
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipeline->pipelines[0]);

            // Draw rectangle
            commandBuffer.bindVertexBuffers(0, rectangle->buffer(), {0});
            commandBuffer.bindIndexBuffer(rectangle->buffer(), rectangle->offset(1), vk::IndexType::eUint16);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, this->pipeline->layouts[0], 0,
                                             this->descriptorSets[(i * this->swapchain->buffers.size()) + frameIndex], {});

            commandBuffer.drawIndexed(static_cast<u32>(this->indices.size()), 1, 0, 0, 0);
        }
        commandBuffer.end();

        // Pass to primary
        primaryCmd.executeCommands(commandBuffer);
    }
}

void RectangleDemo::beforeCommandBuffersUpdate() {
    if (!this->clockInit) {
        this->clock = std::chrono::system_clock::now();
        this->clockInit = true;

        // Generate vector of rotations
        std::srand(std::time(nullptr));
        for (size_t i = 0; i < RECTANGLE_COUNT; i++) {
            this->rotations.push_back(std::make_pair(static_cast<float>((std::rand() % (180 - 10)) + 10), glm::vec3(0.f, 1.0f, 1.0f)));
        }

        return;
    }

    // Delta time
    float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::system_clock::now() - this->clock).count();

    // Update uniform buffers
    for (size_t i = 0; i < RECTANGLE_COUNT; i++) {
        this->_uniformBuffers[(i * this->swapchain->buffers.size()) + this->swapchain->frame_index].model =
            glm::rotate(glm::mat4(1.0f), deltaTime * glm::radians(this->rotations[i].first), this->rotations[i].second);
        this->_uniformBuffers[(i * this->swapchain->buffers.size()) + this->swapchain->frame_index].view =
            glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        this->_uniformBuffers[(i * this->swapchain->buffers.size()) + this->swapchain->frame_index].proj = glm::perspective(
            glm::radians(45.0f), this->swapchain->current_extent.width / static_cast<float>(this->swapchain->current_extent.height), 0.1f, 10.0f);
        this->_uniformBuffers[(i * this->swapchain->buffers.size()) + this->swapchain->frame_index].proj[1][1] *= -1;  // Adapt for vulkan

        // Update buffer
        this->uniformBuffer->updateFragment((i * this->swapchain->buffers.size()) + this->swapchain->frame_index,
                                            &this->_uniformBuffers[(i * this->swapchain->buffers.size()) + this->swapchain->frame_index]);
    }
}

vk::QueueFlags RectangleDemo::queueFlags() const {
    return ao::vulkan::GLFWEngine::queueFlags() | vk::QueueFlagBits::eTransfer;  // Enable transfer
}

void RectangleDemo::createDescriptorSetLayouts() {
    // Create binding
    vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);

    // Create info
    vk::DescriptorSetLayoutCreateInfo createInfo(vk::DescriptorSetLayoutCreateFlags(), 1, &binding);

    // Create layouts
    for (size_t i = 0; i < this->swapchain->buffers.size() * RECTANGLE_COUNT; i++) {
        this->descriptorSetLayouts.push_back(this->device->logical.createDescriptorSetLayout(createInfo));
    }
}

void RectangleDemo::createDescriptorPools() {
    vk::DescriptorPoolSize poolSize(vk::DescriptorType::eUniformBuffer, static_cast<u32>(this->swapchain->buffers.size()));

    // Create pool
    this->descriptorPools.push_back(this->device->logical.createDescriptorPool(vk::DescriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->buffers.size() * RECTANGLE_COUNT), 1, &poolSize)));
}

void RectangleDemo::createDescriptorSets() {
    vk::DescriptorSetAllocateInfo allocateInfo(this->descriptorPools[0], static_cast<u32>(this->swapchain->buffers.size() * RECTANGLE_COUNT),
                                               this->descriptorSetLayouts.data());

    // Create sets
    this->descriptorSets = this->device->logical.allocateDescriptorSets(allocateInfo);

    // Configure
    for (size_t i = 0; i < this->swapchain->buffers.size() * RECTANGLE_COUNT; i++) {
        vk::DescriptorBufferInfo bufferInfo(this->uniformBuffer->buffer(), this->uniformBuffer->offset(i), sizeof(UniformBufferObject));
        this->device->logical.updateDescriptorSets(
            vk::WriteDescriptorSet(this->descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo), {});
    }
}
