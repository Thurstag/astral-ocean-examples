// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "texture_array.h"

#include <ao/vulkan/wrapper/pipeline/graphics_pipeline.h>
#include <boost/filesystem.hpp>
#include <gli/gli.hpp>

void TextureArrayDemo::onKeyEventCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    ao::vulkan::GLFWEngine::onKeyEventCallback(window, key, scancode, action, mods);

    if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {  // DOWN
        this->array_level_index--;
        if (this->array_level_index > this->array_levels) {
            this->array_level_index = 0;
        }

        this->LOGGER << ao::core::Logger::Level::debug << fmt::format("Base array index: {}", array_level_index);
    } else if (key == GLFW_KEY_UP && action == GLFW_PRESS) {  // UP
        this->array_level_index++;
        this->array_level_index = std::min<u32>(this->array_level_index, this->array_levels - INSTANCE_COUNT);

        this->LOGGER << ao::core::Logger::Level::debug << fmt::format("Base array index: {}", array_level_index);
    }
}

void TextureArrayDemo::freeVulkan() {
    this->model_buffer.reset();
    this->ubo_buffer.reset();

    this->device->logical.destroySampler(this->texture_sampler);

    this->device->logical.destroyImage(std::get<0>(this->texture));
    this->device->logical.destroyImageView(std::get<2>(this->texture));
    this->device->logical.freeMemory(std::get<1>(this->texture));

    ao::vulkan::GLFWEngine::freeVulkan();
}

vk::RenderPass TextureArrayDemo::createRenderPass() {
    // Define attachments
    std::array<vk::AttachmentDescription, 2> attachments;
    attachments[0]
        .setFormat(this->swapchain->colorFormat())
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
    return this->device->logical.createRenderPass(
        vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), static_cast<u32>(attachments.size()), attachments.data(), 1, &subpass, 1, &dependency));
}

void TextureArrayDemo::createPipelines() {
    /* PIPELINE LAYOUT PART */

    // Create bindings
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings;
    bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);
    bindings[1] = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);

    // Create layout
    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    descriptor_set_layouts.push_back(this->device->logical.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<u32>(bindings.size()), bindings.data())));

    std::vector<vk::PushConstantRange> push_constants;
    push_constants.push_back(vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0, sizeof(this->array_level_index)));

    auto pipeline_layout = std::make_shared<ao::vulkan::PipelineLayout>(this->device, descriptor_set_layouts, push_constants);

    /* PIPELINE PART */

    // Create shadermodules
    ao::vulkan::ShaderModule module(this->device);

    // Load shaders & get shaderStages
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages =
        module.loadShader(vk::ShaderStageFlagBits::eVertex, "assets/shaders/texture-array/vert.spv")
            .loadShader(vk::ShaderStageFlagBits::eFragment, "assets/shaders/texture-array/frag.spv")
            .shaderStages();

    // Construct the differnent states making up the pipeline

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
    auto vertex_input = TexturedVertex::BindingDescription();

    // Inpute attribute bindings
    auto vertex_attributes = TexturedVertex::AttributeDescriptions();

    // Vertex input state used for pipeline creation
    vk::PipelineVertexInputStateCreateInfo vertex_state(vk::PipelineVertexInputStateCreateFlags(), 1, &vertex_input,
                                                        static_cast<u32>(vertex_attributes.size()), vertex_attributes.data());

    // Cache create info
    auto cache = ao::vulkan::GLFWEngine::LoadCache("data/texture-array/caches/main.cache");
    vk::PipelineCacheCreateInfo cache_info(vk::PipelineCacheCreateFlags(), cache.size(), cache.data());

    // Create rendering pipeline using the specified states
    this->pipelines["main"] = new ao::vulkan::GraphicsPipeline(this->device, pipeline_layout, this->render_pass, shader_stages, vertex_state,
                                                               input_state, std::nullopt, viewport_state, rasterization_state, multisample_state,
                                                               depth_stencil_state, color_state, dynamic_state, cache_info);

    // Define callback
    auto device = this->device;
    this->pipelines.setBeforePipelineCacheDestruction([this, device](std::string name, vk::PipelineCache cache) {
        this->saveCache("data/texture-array/caches", name + std::string(".cache"), cache);
    });

    /* DESCRIPTOR POOL PART */

    std::array<vk::DescriptorPoolSize, 2> poolSizes;
    poolSizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, static_cast<u32>(this->swapchain->size()));
    poolSizes[1] = vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, static_cast<u32>(this->swapchain->size()));

    this->pipelines["main"]->pools().push_back(std::move(ao::vulkan::DescriptorPool(
        this->device, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size()),
                                                   static_cast<u32>(poolSizes.size()), poolSizes.data()))));
}

void TextureArrayDemo::createVulkanBuffers() {
    // Create vertices & indices
    this->model_buffer =
        std::make_unique<ao::vulkan::StagingTupleBuffer<TexturedVertex, u16>>(this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    this->model_buffer->init({sizeof(TexturedVertex) * this->vertices.size(), sizeof(u16) * this->indices.size()})
        ->update(this->vertices.data(), this->indices.data());

    this->model_buffer->freeHostBuffer();

    this->ubo_buffer = std::make_unique<ao::vulkan::BasicDynamicArrayBuffer<UBO>>(this->swapchain->size(), this->device);
    this->ubo_buffer
        ->init(vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive, vk::MemoryPropertyFlagBits::eHostVisible,
               ao::vulkan::Buffer::CalculateUBOAligmentSize(this->device->physical, sizeof(UBO)))
        ->map();

    // Resize uniform buffers vector
    this->uniform_buffers.resize(this->swapchain->size());

    /* TEXTURE CREATION */

    // Load texture
    char* texture_file = "assets/textures/texturearray.ktx";
    if (!boost::filesystem::exists(texture_file)) {
        throw ao::core::Exception(fmt::format("{} doesn't exist", texture_file));
    }
    gli::texture2d_array texture_image(gli::load(texture_file));

    // Check image
    if (texture_image.empty()) {
        throw ao::core::Exception(fmt::format("Fail to load image: {0}", texture_file));
    }
    this->array_levels = static_cast<u32>(texture_image.layers());
    auto image_format = vk::Format(texture_image.format());  // Convert format

    this->LOGGER << ao::core::Logger::Level::debug << fmt::format("Load texture with format: {}", vk::to_string(image_format));

    // Create buffer
    auto textureBuffer = ao::vulkan::BasicTupleBuffer<u8>(this->device);
    textureBuffer
        .init(vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
              vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible, {texture_image.size()})
        ->update(static_cast<u8*>(texture_image.data()));

    // Create image
    auto image = this->device->createImage(
        texture_image.extent().x, texture_image.extent().y, 1, this->array_levels, image_format, vk::ImageType::e2D, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Assign
    std::get<0>(this->texture) = image.first;
    std::get<1>(this->texture) = image.second;

    // Create BufferCopy
    std::vector<vk::BufferImageCopy> regions(array_levels);
    vk::DeviceSize offset = 0;
    for (size_t i = 0; i < array_levels; i++) {
        regions[i] = vk::BufferImageCopy(offset, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, static_cast<u32>(i), 1),
                                         vk::Offset3D(), vk::Extent3D(texture_image[i][0].extent().x, texture_image[i][0].extent().x, 1));

        offset += texture_image[i][0].size();
    }

    // Process image & copy into image
    this->device->processImage(std::get<0>(this->texture), image_format,
                               vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, this->array_levels), vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eTransferDstOptimal);
    this->device->copyBufferToImage(textureBuffer.buffer(), std::get<0>(this->texture), regions);
    this->device->processImage(std::get<0>(this->texture), image_format,
                               vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, this->array_levels),
                               vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create view
    std::get<2>(this->texture) =
        this->device->createImageView(std::get<0>(this->texture), image_format, vk::ImageViewType::e2DArray,
                                      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, this->array_levels));

    // Create sampler
    this->texture_sampler = this->device->logical.createSampler(
        vk::SamplerCreateInfo(vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
                              vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, 0,
                              VK_TRUE, 16, VK_FALSE, vk::CompareOp::eAlways, 0, 0, vk::BorderColor::eFloatOpaqueBlack, VK_FALSE));

    /* DESCRIPTOR SETS CREATION */

    // Create vector of layouts
    std::vector<vk::DescriptorSetLayout> layouts(this->swapchain->size(), this->pipelines["main"]->layout()->descriptorLayouts().front());

    // Create sets
    auto descriptor_sets = this->pipelines["main"]->pools().front().allocateDescriptorSets(static_cast<u32>(this->swapchain->size()), layouts);

    // Configure
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        vk::DescriptorBufferInfo bufferInfo(this->ubo_buffer->buffer(), this->ubo_buffer->offset(i), sizeof(UBO));
        this->device->logical.updateDescriptorSets(
            vk::WriteDescriptorSet(descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo), {});

        vk::DescriptorImageInfo sampleInfo(this->texture_sampler, std::get<2>(this->texture), vk::ImageLayout::eShaderReadOnlyOptimal);
        this->device->logical.updateDescriptorSets(
            vk::WriteDescriptorSet(descriptor_sets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &sampleInfo), {});
    }
}

void TextureArrayDemo::createSecondaryCommandBuffers() {
    this->command_buffers =
        this->secondary_command_pool->allocateCommandBuffers(vk::CommandBufferLevel::eSecondary, static_cast<u32>(this->swapchain->size()));
}

void TextureArrayDemo::executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritanceInfo, int frameIndex,
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

        // Push constants
        commandBuffer.pushConstants<u32>(this->pipelines["main"]->layout()->value(), vk::ShaderStageFlagBits::eVertex, 0, this->array_level_index);

        // Bind pipeline
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipelines["main"]->value());

        // Draw rectangle
        commandBuffer.bindVertexBuffers(0, rectangle->buffer(), {0});
        commandBuffer.bindIndexBuffer(rectangle->buffer(), rectangle->offset(1), vk::IndexType::eUint16);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, this->pipelines["main"]->layout()->value(), 0,
                                         this->pipelines["main"]->pools().front().descriptorSets().at(frameIndex), {});

        commandBuffer.drawIndexed(static_cast<u32>(this->indices.size()), INSTANCE_COUNT, 0, 0, 0);
    }
    commandBuffer.end();

    // Pass to primary
    primaryCmd.executeCommands(commandBuffer);
}

void TextureArrayDemo::beforeCommandBuffersUpdate() {
    if (!this->clock_start) {
        this->clock = std::chrono::system_clock::now();
        this->clock_start = true;

        return;
    }

    // Update uniform buffer
    for (size_t i = 0; i < INSTANCE_COUNT; i++) {
        this->uniform_buffers[this->swapchain->currentFrameIndex()].instances[i].rotation =
            glm::rotate(glm::mat4(1.0f), glm::radians(360.0f), glm::vec3(.0f, 0.0f, 1.0f));
        this->uniform_buffers[this->swapchain->currentFrameIndex()].instances[i].positionAndScale =
            glm::vec4(.0f, .0f, (i % 2 == 0 ? -.5f : .5f) * i, 1.0f);
    }

    this->uniform_buffers[this->swapchain->currentFrameIndex()].view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    this->uniform_buffers[this->swapchain->currentFrameIndex()].proj =
        glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / (float)this->swapchain->extent().height, 0.1f, 10.0f);
    this->uniform_buffers[this->swapchain->currentFrameIndex()].proj[1][1] *= -1;  // Adapt for vulkan

    // Update buffer
    this->ubo_buffer->updateFragment(this->swapchain->currentFrameIndex(), &this->uniform_buffers[this->swapchain->currentFrameIndex()]);
}
