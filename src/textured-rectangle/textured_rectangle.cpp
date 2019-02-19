// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "textured_rectangle.h"

#define STB_IMAGE_IMPLEMENTATION
#include <ao/vulkan/wrapper/pipeline/graphics_pipeline.h>
#include <stb_image.h>

void TexturedRectangle::freeVulkan() {
    this->model_buffer.reset();
    this->ubo_buffer.reset();

    this->device->logical.destroySampler(this->texture_sampler);

    this->device->logical.destroyImage(std::get<0>(this->texture));
    this->device->logical.destroyImageView(std::get<2>(this->texture));
    this->device->logical.freeMemory(std::get<1>(this->texture));

    ao::vulkan::GLFWEngine::freeVulkan();
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
    vk::AttachmentReference color_ref = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpass =
        vk::SubpassDescription().setPipelineBindPoint(vk::PipelineBindPoint::eGraphics).setColorAttachmentCount(1).setPColorAttachments(&color_ref);
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

void TexturedRectangle::createPipelines() {
    /* PIPELINE LAYOUT PART */

    // Create bindings
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings;
    bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);
    bindings[1] = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);

    // Create layout
    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    descriptor_set_layouts.push_back(this->device->logical.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<u32>(bindings.size()), bindings.data())));

    auto pipeline_layout = std::make_shared<ao::vulkan::PipelineLayout>(this->device, descriptor_set_layouts);

    /* PIPELINE PART */

    // Create shadermodules
    ao::vulkan::ShaderModule module(this->device);

    // Load shaders & get shaderStages
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages =
        module.loadShader(vk::ShaderStageFlagBits::eVertex, "assets/shaders/textured-rectangle/vert.spv")
            .loadShader(vk::ShaderStageFlagBits::eFragment, "assets/shaders/textured-rectangle/frag.spv")
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
    vk::PipelineDepthStencilStateCreateInfo depth_stencil_state;

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
    auto cache = ao::vulkan::GLFWEngine::LoadCache("data/textured-rectangle/caches/main.cache");
    vk::PipelineCacheCreateInfo cache_info(vk::PipelineCacheCreateFlags(), cache.size(), cache.data());

    // Create rendering pipeline using the specified states
    this->pipelines["main"] = new ao::vulkan::GraphicsPipeline(this->device, pipeline_layout, this->render_pass, shader_stages, vertex_state,
                                                               input_state, std::nullopt, viewport_state, rasterization_state, multisample_state,
                                                               depth_stencil_state, color_state, dynamic_state, cache_info);

    // Define callback
    auto device = this->device;
    this->pipelines.setBeforePipelineCacheDestruction([this, device](std::string name, vk::PipelineCache cache) {
        this->saveCache("data/textured-rectangle/caches", name + std::string(".cache"), cache);
    });

    /* DESCRIPTOR POOL PART */

    std::array<vk::DescriptorPoolSize, 2> pool_sizes;
    pool_sizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, static_cast<u32>(this->swapchain->size()));
    pool_sizes[1] = vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, static_cast<u32>(this->swapchain->size()));

    this->pipelines["main"]->pools().push_back(std::move(ao::vulkan::DescriptorPool(
        this->device, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size()),
                                                   static_cast<u32>(pool_sizes.size()), pool_sizes.data()))));
}

void TexturedRectangle::createVulkanBuffers() {
    // Create vertices & indices
    this->model_buffer =
        std::make_unique<ao::vulkan::StagingTupleBuffer<TexturedVertex, u16>>(this->device, vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    this->model_buffer->init({sizeof(TexturedVertex) * this->vertices.size(), sizeof(u16) * this->indices.size()})
        ->update(this->vertices.data(), this->indices.data());

    this->model_buffer->freeHostBuffer();

    this->ubo_buffer = std::make_unique<ao::vulkan::BasicDynamicArrayBuffer<UniformBufferObject>>(this->swapchain->size(), this->device);
    this->ubo_buffer->init(vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive, vk::MemoryPropertyFlagBits::eHostVisible,
                           ao::vulkan::Buffer::CalculateUBOAligmentSize(this->device->physical, sizeof(UniformBufferObject)));

    // Map buffer
    this->ubo_buffer->map();

    // Resize uniform buffers vector
    this->uniform_buffers.resize(this->swapchain->size());

    /* TEXTURE CREATION */

    // Load texture
    char* texture_file = "assets/textures/face.jpg";
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
    auto image =
        this->device->createImage(texture_width, texture_height, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageType::e2D, vk::ImageTiling::eOptimal,
                                  vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Assign
    std::get<0>(this->texture) = image.first;
    std::get<1>(this->texture) = image.second;

    // Process image & copy into image
    this->device->processImage(std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm,
                               vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eTransferDstOptimal);
    this->device->copyBufferToImage(texture_buffer.buffer(), std::get<0>(this->texture),
                                    vk::BufferImageCopy(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), vk::Offset3D(),
                                                        vk::Extent3D(vk::Extent2D(texture_width, texture_height), 1)));
    this->device->processImage(std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm,
                               vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal,
                               vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create view
    std::get<2>(this->texture) = this->device->createImageView(std::get<0>(this->texture), vk::Format::eR8G8B8A8Unorm, vk::ImageViewType::e2D,
                                                               vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

    // Create sampler
    this->texture_sampler = this->device->logical.createSampler(
        vk::SamplerCreateInfo(vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
                              vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0, VK_TRUE, 16,
                              VK_FALSE, vk::CompareOp::eAlways, 0, 0, vk::BorderColor::eFloatOpaqueBlack, VK_FALSE));

    /* DESCRIPTOR SETS CREATION */

    // Create vector of layouts
    std::vector<vk::DescriptorSetLayout> layouts(this->swapchain->size(), this->pipelines["main"]->layout()->descriptorLayouts().front());

    // Create sets
    auto descriptor_sets = this->pipelines["main"]->pools().front().allocateDescriptorSets(static_cast<u32>(this->swapchain->size()), layouts);

    // Configure
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        vk::DescriptorImageInfo sample_info(this->texture_sampler, std::get<2>(this->texture), vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorBufferInfo buffer_info(this->ubo_buffer->buffer(), this->ubo_buffer->offset(i), sizeof(UniformBufferObject));

        this->device->logical.updateDescriptorSets(
            vk::WriteDescriptorSet(descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info), {});

        this->device->logical.updateDescriptorSets(
            vk::WriteDescriptorSet(descriptor_sets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &sample_info), {});
    }
}

void TexturedRectangle::createSecondaryCommandBuffers() {
    this->command_buffers =
        this->secondary_command_pool->allocateCommandBuffers(vk::CommandBufferLevel::eSecondary, static_cast<u32>(this->swapchain->size()));

    for (auto& command_buffer : this->command_buffers) {
        this->to_update[command_buffer] = true;
    }
}

void TexturedRectangle::executeSecondaryCommandBuffers(vk::CommandBufferInheritanceInfo& inheritance_info, int frame_index,
                                                       vk::CommandBuffer primary_command) {
    auto& command_buffer = this->command_buffers[frame_index];

    // Reset all command buffers
    if (this->swapchain->state() == ao::vulkan::SwapchainState::eReset) {
        for (auto [key, value] : this->to_update) {
            this->to_update[key] = true;
        }
    }

    // Draw in command
    if (this->to_update[command_buffer]) {
        // Begin info
        vk::CommandBufferBeginInfo begin_info =
            vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eRenderPassContinue).setPInheritanceInfo(&inheritance_info);

        command_buffer.begin(begin_info);
        {
            // Set viewport & scissor
            command_buffer.setViewport(0, vk::Viewport(0, 0, static_cast<float>(this->swapchain->extent().width),
                                                       static_cast<float>(this->swapchain->extent().height), 0, 1));
            command_buffer.setScissor(0, vk::Rect2D(vk::Offset2D(), this->swapchain->extent()));

            // Bind pipeline
            command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipelines["main"]->value());

            // Draw rectangle
            ao::vulkan::TupleBuffer<TexturedVertex, u16>* rectangle = this->model_buffer.get();
            command_buffer.bindVertexBuffers(0, rectangle->buffer(), {0});
            command_buffer.bindIndexBuffer(rectangle->buffer(), rectangle->offset(1), vk::IndexType::eUint16);
            command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, this->pipelines["main"]->layout()->value(), 0,
                                              this->pipelines["main"]->pools().front().descriptorSets().at(frame_index), {});

            command_buffer.drawIndexed(static_cast<u32>(this->indices.size()), 1, 0, 0, 0);
        }
        command_buffer.end();

        this->to_update[command_buffer] = false;
    }

    // Pass to primary
    primary_command.executeCommands(command_buffer);
}

void TexturedRectangle::beforeCommandBuffersUpdate() {
    if (!this->clock_start) {
        this->clock = std::chrono::system_clock::now();
        this->clock_start = true;

        // Init uniform buffers
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            this->uniform_buffers[i].view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        }
        return;
    }

    // Delta time
    float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::system_clock::now() - this->clock).count();

    // Update uniform buffer
    this->uniform_buffers[this->swapchain->currentFrameIndex()].rotation =
        glm::rotate(glm::mat4(1.0f), delta_time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    this->uniform_buffers[this->swapchain->currentFrameIndex()].proj =
        glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / (float)this->swapchain->extent().height, 0.1f, 10.0f);
    this->uniform_buffers[this->swapchain->currentFrameIndex()].proj[1][1] *= -1;  // Adapt for vulkan

    // Update buffer
    this->ubo_buffer->updateFragment(this->swapchain->currentFrameIndex(), &this->uniform_buffers[this->swapchain->currentFrameIndex()]);
}
