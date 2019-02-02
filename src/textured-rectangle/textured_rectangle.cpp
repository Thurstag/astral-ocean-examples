// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "textured_rectangle.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

TexturedRectangle::~TexturedRectangle() {
    this->model_buffer.reset();
    this->ubo_buffer.reset();

    this->device->logical.destroySampler(this->texture_sampler);

    this->device->logical.destroyImage(std::get<0>(this->texture));
    this->device->logical.destroyImageView(std::get<2>(this->texture));
    this->device->logical.freeMemory(std::get<1>(this->texture));
}

vk::RenderPass TexturedRectangle::createRenderPass() {
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

void TexturedRectangle::createPipelineLayouts() {
    this->pipeline->layouts.resize(1);

    this->pipeline->layouts.front() = this->device->logical.createPipelineLayout(vk::PipelineLayoutCreateInfo(
        vk::PipelineLayoutCreateFlags(), static_cast<u32>(this->descriptorSetLayouts.size()), this->descriptorSetLayouts.data()));
}

void TexturedRectangle::setUpPipelines() {
    // Create shadermodules
    ao::vulkan::ShaderModule module(this->device);

    // Load shaders & get shaderStages
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages =
        module.loadShader(vk::ShaderStageFlagBits::eVertex, "assets/shaders/textured-rectangle/vert.spv")
            .loadShader(vk::ShaderStageFlagBits::eFragment, "assets/shaders/textured-rectangle/frag.spv")
            .shaderStages();

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
        vk::GraphicsPipelineCreateInfo().setLayout(this->pipeline->layouts[0]).setRenderPass(this->render_pass);

    // Construct the differnent states making up the pipeline

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
    vk::Viewport viewport(0, 0, static_cast<float>(this->swapchain->extent().width), static_cast<float>(this->swapchain->extent().height), 0, 1);
    vk::Rect2D scissor(vk::Offset2D(), this->swapchain->extent());
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
    auto vertexInputBinding = TexturedVertex::BindingDescription();

    // Inpute attribute bindings
    auto vertexInputAttributes = TexturedVertex::AttributeDescriptions();

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
        .setRenderPass(this->render_pass)
        .setPDynamicState(&dynamicState);

    // Create rendering pipeline using the specified states
    this->pipeline->pipelines = this->device->logical.createGraphicsPipelines(this->pipeline->cache, pipelineCreateInfo);
}

void TexturedRectangle::createVulkanBuffers() {
    // Create vertices & indices
    this->model_buffer = std::unique_ptr<ao::vulkan::TupleBuffer<TexturedVertex, u16>>(
        (new ao::vulkan::StagingTupleBuffer<TexturedVertex, u16>(this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit))
            ->init({sizeof(TexturedVertex) * this->vertices.size(), sizeof(u16) * this->indices.size()})
            ->update(this->vertices.data(), this->indices.data()));

    this->ubo_buffer = std::unique_ptr<ao::vulkan::DynamicArrayBuffer<UniformBufferObject>>(
        (new ao::vulkan::BasicDynamicArrayBuffer<UniformBufferObject>(this->swapchain->size(), this->device))
            ->init(vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive, vk::MemoryPropertyFlagBits::eHostVisible,
                   ao::vulkan::Buffer::CalculateUBOAligmentSize(this->device->physical, sizeof(UniformBufferObject))));

    // Map buffer
    this->ubo_buffer->map();

    // Resize uniform buffers vector
    this->uniform_buffers.resize(this->swapchain->size());

    /* TEXTURE CREATION */

    // Load texture
    char* textureFile = "assets/textures/face.jpg";
    int texWidth, texHeight, texChannels;
    pixel_t* pixels = stbi_load(textureFile, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    // Check image
    if (!pixels) {
        throw ao::core::Exception(fmt::format("Fail to load image: {0}", textureFile));
    }

    // Create buffer
    auto textureBuffer = ao::vulkan::BasicTupleBuffer<pixel_t>(this->device);
    textureBuffer
        .init(vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
              vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible, {texWidth * texHeight * 4 * sizeof(pixel_t)})
        ->update(pixels);

    auto a = texWidth * texHeight * 4;

    // Free image
    stbi_image_free(pixels);

    // Create image
    auto image =
        this->device->createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Unorm, vk::ImageType::e2D, vk::ImageTiling::eOptimal,
                                  vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Assign
    std::get<0>(this->texture) = image.first;
    std::get<1>(this->texture) = image.second;

    // Process image & copy into image
    this->device->processImage(std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eTransferDstOptimal);
    this->device->copyBufferToImage(textureBuffer.buffer(), std::get<0>(this->texture), texWidth, texHeight);
    this->device->processImage(std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal,
                               vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create view
    std::get<2>(this->texture) = this->device->createImageView(std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm, vk::ImageViewType::e2D,
                                                               vk::ImageAspectFlagBits::eColor);

    // Create sampler
    this->texture_sampler = this->device->logical.createSampler(
        vk::SamplerCreateInfo(vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
                              vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0, VK_TRUE, 16,
                              VK_FALSE, vk::CompareOp::eAlways, 0, 0, vk::BorderColor::eFloatOpaqueBlack, VK_FALSE));
}

void TexturedRectangle::createSecondaryCommandBuffers() {
    this->command_buffers = this->secondary_command_pool->allocateCommandBuffers(vk::CommandBufferLevel::eSecondary, this->swapchain->size());
}

void TexturedRectangle::executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritanceInfo, int frameIndex,
                                                       vk::CommandBuffer primaryCmd) {
    // Create info
    vk::CommandBufferBeginInfo beginInfo =
        vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue).setPInheritanceInfo(&inheritanceInfo);
    ao::vulkan::TupleBuffer<TexturedVertex, u16>* rectangle = this->model_buffer.get();

    // Draw in command
    auto& commandBuffer = this->command_buffers[frameIndex];
    commandBuffer.begin(beginInfo);
    {
        // Set viewport & scissor
        commandBuffer.setViewport(
            0, vk::Viewport(0, 0, static_cast<float>(this->swapchain->extent().width), static_cast<float>(this->swapchain->extent().height), 0, 1));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(), this->swapchain->extent()));

        // Bind pipeline
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipeline->pipelines[0]);

        // Draw rectangle
        commandBuffer.bindVertexBuffers(0, rectangle->buffer(), {0});
        commandBuffer.bindIndexBuffer(rectangle->buffer(), rectangle->offset(1), vk::IndexType::eUint16);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, this->pipeline->layouts[0], 0, this->descriptorSets[frameIndex], {});

        commandBuffer.drawIndexed(static_cast<u32>(this->indices.size()), 1, 0, 0, 0);
    }
    commandBuffer.end();

    // Pass to primary
    primaryCmd.executeCommands(commandBuffer);
}

void TexturedRectangle::beforeCommandBuffersUpdate() {
    if (!this->clock_start) {
        this->clock = std::chrono::system_clock::now();
        this->clock_start = true;

        return;
    }

    // Delta time
    float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::system_clock::now() - this->clock).count();

    // Update uniform buffer
    this->uniform_buffers[this->swapchain->currentFrameIndex()].model =
        glm::rotate(glm::mat4(1.0f), deltaTime * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    this->uniform_buffers[this->swapchain->currentFrameIndex()].view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    this->uniform_buffers[this->swapchain->currentFrameIndex()].proj =
        glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / (float)this->swapchain->extent().height, 0.1f, 10.0f);
    this->uniform_buffers[this->swapchain->currentFrameIndex()].proj[1][1] *= -1;  // Adapt for vulkan

    // Update buffer
    this->ubo_buffer->updateFragment(this->swapchain->currentFrameIndex(), &this->uniform_buffers[this->swapchain->currentFrameIndex()]);
}

void TexturedRectangle::createDescriptorSetLayouts() {
    // Create bindings
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings;
    bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);
    bindings[1] = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);

    // Create info
    vk::DescriptorSetLayoutCreateInfo createInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<u32>(bindings.size()), bindings.data());

    // Create layouts
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        this->descriptorSetLayouts.push_back(this->device->logical.createDescriptorSetLayout(createInfo));
    }
}

void TexturedRectangle::createDescriptorPools() {
    std::array<vk::DescriptorPoolSize, 2> poolSizes;
    poolSizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, static_cast<u32>(this->swapchain->size()));
    poolSizes[1] = vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, static_cast<u32>(this->swapchain->size()));

    // Create pool
    this->descriptorPools.push_back(this->device->logical.createDescriptorPool(vk::DescriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size()), static_cast<u32>(poolSizes.size()), poolSizes.data())));
}

void TexturedRectangle::createDescriptorSets() {
    vk::DescriptorSetAllocateInfo allocateInfo(this->descriptorPools[0], static_cast<u32>(this->swapchain->size()),
                                               this->descriptorSetLayouts.data());

    // Create sets
    this->descriptorSets = this->device->logical.allocateDescriptorSets(allocateInfo);

    // Configure
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        vk::DescriptorBufferInfo bufferInfo(this->ubo_buffer->buffer(), this->ubo_buffer->offset(i), sizeof(UniformBufferObject));
        this->device->logical.updateDescriptorSets(
            vk::WriteDescriptorSet(this->descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo), {});

        vk::DescriptorImageInfo sampleInfo(this->texture_sampler, std::get<2>(this->texture), vk::ImageLayout::eShaderReadOnlyOptimal);
        this->device->logical.updateDescriptorSets(
            vk::WriteDescriptorSet(this->descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &sampleInfo), {});
    }
}
