// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "mipmap.h"

#include <execution>

#include <ao/vulkan/pipeline/graphics_pipeline.h>
#include <ao/vulkan/utilities/device.h>
#include <meshoptimizer.h>
#include <objparser.h>
#include <boost/filesystem.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/range/irange.hpp>
#include <gli/gli.hpp>

static constexpr char const* MipMapKey = "mimap.enable";

void MipmapDemo::setUpTexture() {
    /* TEXTURE CREATION */

    // Load texture
    char* texture_file = "assets/textures/chalet.ktx";
    if (!boost::filesystem::exists(texture_file)) {
        throw ao::core::Exception(fmt::format("{} doesn't exist", texture_file));
    }
    gli::texture2d texture_image(gli::load(texture_file));

    // Check image
    if (texture_image.empty()) {
        throw ao::core::Exception(fmt::format("Fail to load image: {0}", texture_file));
    }
    auto image_format = vk::Format(texture_image.format());  // Convert format

    LOG_MSG(debug) << fmt::format("Load texture with format: {}", vk::to_string(image_format));

    // Mipmap texture
    {
        auto mip_levels = static_cast<u32>(texture_image.levels());

        // Create buffer
        auto texture_buffer = ao::vulkan::Vector<u8>(texture_image.size() / sizeof(u8), this->host_allocator, vk::BufferUsageFlagBits::eTransferSrc);
        std::copy(std::execution::par_unseq, static_cast<u8*>(texture_image.data()), static_cast<u8*>(texture_image.data()) + texture_buffer.size(),
                  &texture_buffer.at(0));
        texture_buffer.invalidate(0, texture_buffer.size());

        // Create image
        auto image = ao::vulkan::utilities::createImage(
            *this->device->logical(), this->device->physical(), texture_image.extent().x, texture_image.extent().y, mip_levels, 1, image_format,
            vk::ImageType::e2D, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Assign
        std::get<0>(this->mipmap_texture) = image.first;
        std::get<1>(this->mipmap_texture) = image.second;

        // Create regions
        std::vector<vk::BufferImageCopy> regions(mip_levels);
        u32 region_offset = 0;
        for (u32 i = 0; i < mip_levels; i++) {
            regions[i]
                .setBufferOffset(region_offset)
                .setImageExtent(vk::Extent3D(texture_image[i].extent().x, texture_image[i].extent().y, 1))
                .setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, static_cast<u32>(i), 0, 1));

            region_offset += static_cast<u32>(texture_image[i].size());
        }

        // Process image & copy into image
        ao::vulkan::utilities::updateImageLayout(*this->device->logical(), this->device->graphicsPool(), *this->device->queues(),
                                                 std::get<0>(this->mipmap_texture), image_format,
                                                 vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1),
                                                 vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        ao::vulkan::utilities::copyBufferToImage(*this->device->logical(), this->device->transferPool(), *this->device->queues(),
                                                 texture_buffer.info().buffer, std::get<0>(this->mipmap_texture), regions);
        ao::vulkan::utilities::updateImageLayout(*this->device->logical(), this->device->graphicsPool(), *this->device->queues(),
                                                 std::get<0>(this->mipmap_texture), image_format,
                                                 vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1),
                                                 vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        // Create view
        std::get<2>(this->mipmap_texture) =
            ao::vulkan::utilities::createImageView(*this->device->logical(), std::get<0>(this->mipmap_texture), image_format, vk::ImageViewType::e2D,
                                                   vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1));

        // Create sampler
        this->mipmap_texture_sampler = this->device->logical()->createSampler(
            vk::SamplerCreateInfo(vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
                                  vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0, VK_TRUE, 16,
                                  VK_FALSE, vk::CompareOp::eAlways, 0, static_cast<float>(mip_levels), vk::BorderColor::eFloatOpaqueBlack, VK_FALSE));
    }

    // Basic texture
    {
        static constexpr auto mip_levels = 1;

        // Create buffer
        auto texture_buffer =
            ao::vulkan::Vector<u8>(texture_image[0].size() / sizeof(u8), this->host_allocator, vk::BufferUsageFlagBits::eTransferSrc);
        std::copy(std::execution::par_unseq, static_cast<u8*>(texture_image.data()), static_cast<u8*>(texture_image.data()) + texture_buffer.size(),
                  &texture_buffer.at(0));
        texture_buffer.invalidate(0, texture_buffer.size());

        // Create image
        auto image = ao::vulkan::utilities::createImage(
            *this->device->logical(), this->device->physical(), texture_image.extent().x, texture_image.extent().y, mip_levels, 1, image_format,
            vk::ImageType::e2D, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Assign
        std::get<0>(this->texture) = image.first;
        std::get<1>(this->texture) = image.second;

        // Create regions
        std::vector<vk::BufferImageCopy> regions(mip_levels);
        u32 region_offset = 0;
        for (u32 i = 0; i < mip_levels; i++) {
            regions[i]
                .setBufferOffset(region_offset)
                .setImageExtent(vk::Extent3D(texture_image[i].extent().x, texture_image[i].extent().y, 1))
                .setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, static_cast<u32>(i), 0, 1));

            region_offset += static_cast<u32>(texture_image[i].size());
        }

        // Process image & copy into image
        ao::vulkan::utilities::updateImageLayout(*this->device->logical(), this->device->graphicsPool(), *this->device->queues(),
                                                 std::get<0>(this->texture), image_format,
                                                 vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1),
                                                 vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        ao::vulkan::utilities::copyBufferToImage(*this->device->logical(), this->device->transferPool(), *this->device->queues(),
                                                 texture_buffer.info().buffer, std::get<0>(this->texture), regions);
        ao::vulkan::utilities::updateImageLayout(*this->device->logical(), this->device->graphicsPool(), *this->device->queues(),
                                                 std::get<0>(this->texture), image_format,
                                                 vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1),
                                                 vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        // Create view
        std::get<2>(this->texture) =
            ao::vulkan::utilities::createImageView(*this->device->logical(), std::get<0>(this->texture), image_format, vk::ImageViewType::e2D,
                                                   vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 1));

        // Create sampler
        this->texture_sampler = this->device->logical()->createSampler(
            vk::SamplerCreateInfo(vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
                                  vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0, VK_TRUE, 16,
                                  VK_FALSE, vk::CompareOp::eAlways, 0, static_cast<float>(mip_levels), vk::BorderColor::eFloatOpaqueBlack, VK_FALSE));
    }

    /* DESCRIPTOR SETS CREATION */

    // Create vector of layouts
    std::vector<vk::DescriptorSetLayout> layouts(this->swapchain->size(), this->pipelines["main"]->layout()->descriptorLayouts().front());

    // Create sets
    std::vector<vk::DescriptorSet> descriptor_sets =
        this->pipelines["main"]->pools().front().allocateDescriptorSets(static_cast<u32>(this->swapchain->size()), layouts);

    // Configure
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        vk::DescriptorImageInfo mipmap_sample_info(this->mipmap_texture_sampler, std::get<2>(this->mipmap_texture),
                                                   vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo sample_info(this->texture_sampler, std::get<2>(this->texture), vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorBufferInfo buffer_info(this->ubo_buffer->info().buffer, this->ubo_buffer->offset(i), sizeof(UniformBufferObject));

        this->device->logical()->updateDescriptorSets(
            vk::WriteDescriptorSet(descriptor_sets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info), {});

        this->device->logical()->updateDescriptorSets(
            vk::WriteDescriptorSet(descriptor_sets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &sample_info), {});

        this->device->logical()->updateDescriptorSets(
            vk::WriteDescriptorSet(descriptor_sets[i], 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &mipmap_sample_info), {});
    }
}

void MipmapDemo::initVulkan() {
    ao::vulkan::GLFWEngine::initVulkan();
    static constexpr u16 tick_rate = 500;

    this->scheduler.schedule(tick_rate, [&]() {
        auto toggle_state = glfwGetKey(window, GLFW_KEY_T);

        // Toggle mimap
        if (this->toggle_last_state == GLFW_RELEASE && toggle_state == GLFW_PRESS) {
            this->settings_->get<bool>(MipMapKey) = !this->settings_->get<bool>(MipMapKey);

            LOG_MSG(debug) << fmt::format("Mimap: {}", this->settings_->get<bool>(MipMapKey) ? "On" : "Off");

            // Invalidate command buffers
            for (size_t i = 0; i < this->primary_command_buffers.size(); i++) {
                this->primary_command_buffers[i]->invalidateSecondary();
            }
        }

        // Save state
        this->toggle_last_state = toggle_state;
    });
}

void MipmapDemo::freeVulkan() {
    // Free buffers
    this->model_buffer.reset();
    this->ubo_buffer.reset();

    // Free wrappers
    for (auto buffer : this->secondary_command_buffers) {
        delete buffer;
    }

    this->device->logical()->destroySampler(this->texture_sampler);

    this->device->logical()->destroyImage(std::get<0>(this->texture));
    this->device->logical()->destroyImageView(std::get<2>(this->texture));
    this->device->logical()->freeMemory(std::get<1>(this->texture));

    this->device->logical()->destroySampler(this->mipmap_texture_sampler);

    this->device->logical()->destroyImage(std::get<0>(this->mipmap_texture));
    this->device->logical()->destroyImageView(std::get<2>(this->mipmap_texture));
    this->device->logical()->freeMemory(std::get<1>(this->mipmap_texture));

    ao::vulkan::GLFWEngine::freeVulkan();
}

vk::RenderPass MipmapDemo::createRenderPass() {
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

void MipmapDemo::createPipelines() {
    /* PIPELINE LAYOUT PART */

    // Create bindings
    std::array<vk::DescriptorSetLayoutBinding, 3> bindings;
    bindings[0] = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);
    bindings[1] = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);
    bindings[2] = vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);

    // Create layout
    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    descriptor_set_layouts.push_back(this->device->logical()->createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), static_cast<u32>(bindings.size()), bindings.data())));

    std::vector<vk::PushConstantRange> push_constants;
    push_constants.push_back(vk::PushConstantRange(vk::ShaderStageFlagBits::eFragment, 0, sizeof(u32)));

    auto pipeline_layout = std::make_shared<ao::vulkan::PipelineLayout>(this->device->logical(), descriptor_set_layouts, push_constants);

    /* PIPELINE PART */

    // Create shadermodules
    ao::vulkan::ShaderModule module(this->device->logical());

    // Load shaders & get shaderStages
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages =
        module.loadShader(vk::ShaderStageFlagBits::eVertex, "assets/shaders/model/vert.spv")
            .loadShader(vk::ShaderStageFlagBits::eFragment, "assets/shaders/mimap/frag.spv")
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
    auto vertex_input = TexturedVertex::BindingDescription();

    // Inpute attribute bindings
    auto vertex_attributes = TexturedVertex::AttributeDescriptions();

    // Vertex input state used for pipeline creation
    vk::PipelineVertexInputStateCreateInfo vertex_state(vk::PipelineVertexInputStateCreateFlags(), 1, &vertex_input,
                                                        static_cast<u32>(vertex_attributes.size()), vertex_attributes.data());

    // Cache create info
    auto cache = ao::vulkan::GLFWEngine::LoadCache("data/mipmap/caches/main.cache");
    vk::PipelineCacheCreateInfo cache_info(vk::PipelineCacheCreateFlags(), cache.size(), cache.data());

    // Create rendering pipeline using the specified states
    this->pipelines["main"] = new ao::vulkan::GraphicsPipeline(this->device->logical(), pipeline_layout, this->render_pass, shader_stages,
                                                               vertex_state, input_state, std::nullopt, viewport_state, rasterization_state,
                                                               multisample_state, depth_stencil_state, color_state, dynamic_state, cache_info);

    // Define callback
    this->pipelines.setBeforePipelineCacheDestruction([this, device = this->device](std::string name, vk::PipelineCache cache) {
        this->saveCache("data/mipmap/caches", name + std::string(".cache"), cache);
    });

    /* DESCRIPTOR POOL PART */

    std::array<vk::DescriptorPoolSize, 2> pool_sizes;
    pool_sizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, static_cast<u32>(this->swapchain->size()));
    pool_sizes[1] = vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, static_cast<u32>(this->swapchain->size() * 2));

    this->pipelines["main"]->pools().push_back(std::move(ao::vulkan::DescriptorPool(
        this->device->logical(), vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), static_cast<u32>(this->swapchain->size()),
                                                              static_cast<u32>(pool_sizes.size()), pool_sizes.data()))));
}

void MipmapDemo::createVulkanBuffers() {
    /* LOAD MODEL */

    // Load
    ObjFile model;
    LOG_MSG(trace) << "=== Start loading model ===";
    if (!objParseFile(model, "assets/models/chalet.obj")) {
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

    this->ubo_buffer = std::make_unique<ao::vulkan::Vector<UniformBufferObject>>(this->swapchain->size(), this->host_uniform_allocator,
                                                                                 vk::BufferUsageFlagBits::eUniformBuffer);
    // Load texture...
    this->setUpTexture();
}

void MipmapDemo::createSecondaryCommandBuffers() {
    auto command_buffers =
        this->secondary_command_pool->allocateCommandBuffers(vk::CommandBufferLevel::eSecondary, static_cast<u32>(this->swapchain->size()));

    this->secondary_command_buffers.resize(command_buffers.size());
    for (size_t i = 0; i < command_buffers.size(); i++) {
        this->secondary_command_buffers[i] = new ao::vulkan::GraphicsPrimaryCommandBuffer::SecondaryCommandBuffer(
            command_buffers[i], [pipeline = this->pipelines["main"], indices_count = this->indices_count, vertices_count = this->vertices_count,
                                 model = this->model_buffer.get(), &ubo_buffer = this->ubo_buffer, &settings = this->settings_](
                                    vk::CommandBuffer command_buffer, vk::CommandBufferInheritanceInfo const& inheritance_info,
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

                    // Push constants
                    command_buffer.pushConstants<u32>(pipeline->layout()->value(), vk::ShaderStageFlagBits::eFragment, 0,
                                                      static_cast<u32>(settings->get<bool>(MipMapKey, true)));

                    // Bind pipeline
                    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->value());

                    // Draw model
                    command_buffer.bindVertexBuffers(0, model->info().buffer, {0});
                    command_buffer.bindIndexBuffer(model->info().buffer, model->offset(sizeof(TexturedVertex) * vertices_count),
                                                   vk::IndexType::eUint32);
                    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout()->value(), 0,
                                                      pipeline->pools().front().descriptorSets().at(frame_index), {});

                    command_buffer.drawIndexed(indices_count, 1, 0, 0, 0);
                }
                command_buffer.end();
            });
    }

    // Add to primary
    for (size_t i = 0; i < this->swapchain->size(); i++) {
        this->primary_command_buffers[i]->addSecondary(this->secondary_command_buffers[i]);
    }
}

void MipmapDemo::beforeCommandBuffersUpdate() {
    if (!this->engine_start) {
        this->engine_start = true;

        // Init camera
        std::get<0>(this->camera) = glm::vec3(2.0f, .0f, 1.0f);  // Position
        std::get<1>(this->camera) = .0f;                         // Rotation (Z)
        std::get<2>(this->camera) = .0f;                         // Rotation (X/Y)
        std::get<3>(this->camera) = .0f;                         // Zoom factor

        // Init uniform buffers
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            auto& ubo = this->ubo_buffer->at(i);

            ubo.rotation = glm::rotate(glm::mat4(1.0f), 125.0f * glm::radians(45.0f), glm::vec3(.0f, .0f, 1.0f));
            ubo.proj = glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / static_cast<float>(this->swapchain->extent().height),
                                        0.1f, 10000000.0f);
            ubo.proj[1][1] *= -1;  // Adapt for vulkan
            ubo.scale = 1.0f;
        }
        return;
    }

    /* CAMERA PART */

    // Constants
    static constexpr float RotationTarget = 45.0f * boost::math::constants::pi<float>() / 180;
    float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(std::chrono::system_clock::now() - this->clock).count();

    // Get states
    auto states = std::make_tuple(glfwGetKey(window, GLFW_KEY_RIGHT), glfwGetKey(window, GLFW_KEY_LEFT), glfwGetKey(window, GLFW_KEY_DOWN),
                                  glfwGetKey(window, GLFW_KEY_UP), glfwGetKey(window, GLFW_KEY_PAGE_UP), glfwGetKey(window, GLFW_KEY_PAGE_DOWN));

    // Update camera
    float rotation = .0f;
    glm::vec3 angles = glm::vec3(.0f, .0f, 1.0f);
    if (std::get<GLFW_KEY_LEFT - 262>(states) == GLFW_PRESS) {  // LEFT
        rotation = -delta_time * RotationTarget;
        angles = glm::vec3(.0f, .0f, 1.0f);

        std::get<1>(this->camera) += rotation;
    } else if (std::get<GLFW_KEY_RIGHT - 262>(states) == GLFW_PRESS) {  // RIGHT
        rotation = delta_time * RotationTarget;
        angles = glm::vec3(.0f, .0f, 1.0f);

        std::get<1>(this->camera) += rotation;
    } else if (std::get<GLFW_KEY_UP - 262>(states) == GLFW_PRESS) {  // UP
        if ((std::get<2>(this->camera) - rotation) * (180 / glm::pi<float>()) > -50.f) {
            rotation = -delta_time * RotationTarget;
            angles = glm::vec3(glm::rotate(glm::mat4(1.0f), std::get<1>(this->camera), glm::vec3(.0f, .0f, 1.0f)) * glm::vec4(.0f, 1.0f, .0f, .0f));

            std::get<2>(this->camera) += rotation;
        }
    } else if (std::get<GLFW_KEY_DOWN - 262>(states) == GLFW_PRESS) {  // DOWN
        if ((std::get<2>(this->camera) + rotation) * (180 / glm::pi<float>()) < 50.f) {
            rotation = delta_time * RotationTarget;
            angles = glm::vec3(glm::rotate(glm::mat4(1.0f), std::get<1>(this->camera), glm::vec3(.0f, .0f, 1.0f)) * glm::vec4(.0f, 1.0f, .0f, .0f));

            std::get<2>(this->camera) += rotation;
        }
    } else if (std::get<GLFW_KEY_PAGE_UP - 262>(states) == GLFW_PRESS) {  // ZOOM IN
        std::get<3>(this->camera) = std::min(std::get<3>(this->camera) + .0005f, 0.6f);
    } else if (std::get<GLFW_KEY_PAGE_DOWN - 262>(states) == GLFW_PRESS) {  // ZOOM OUT
        std::get<3>(this->camera) -= .0005f;
    }
    std::get<0>(this->camera) = glm::vec3(glm::rotate(glm::mat4(1.0f), rotation, angles) * glm::vec4(std::get<0>(this->camera), 0.0f));

    // Update uniform buffer
    this->ubo_buffer->at(this->swapchain->frameIndex()).view = glm::lookAt(
        std::get<0>(this->camera) - (std::get<0>(this->camera) * std::get<3>(this->camera)), glm::vec3(.0f, .0f, .0f), glm::vec3(.0f, .0f, 1.0f));

    if (this->swapchain->state() == ao::vulkan::SwapchainState::eReset) {
        for (size_t i = 0; i < this->swapchain->size(); i++) {
            auto& ubo = this->ubo_buffer->at(i);

            ubo.proj = glm::perspective(glm::radians(45.0f), this->swapchain->extent().width / static_cast<float>(this->swapchain->extent().height),
                                        0.1f, 10000000.0f);
            ubo.proj[1][1] *= -1;  // Adapt for vulkan
        }
        this->ubo_buffer->invalidate(0, this->ubo_buffer->size());
    } else {
        this->ubo_buffer->invalidate(this->swapchain->frameIndex());
    }

    // Update clock
    this->clock = std::chrono::system_clock::now();

    // Save states
    this->direction_last_states = states;
}
