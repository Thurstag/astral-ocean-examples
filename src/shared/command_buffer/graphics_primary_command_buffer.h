// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#if defined(__GNUC__) && (__GNUC___ < 9)
#    include "tbb/parallel_for_each.h"
#else
#    include <execution>
#endif

#include "command_buffer.hpp"

namespace ao::vulkan {
    /**
     * @brief Execution policy values
     *
     */
    enum class ExecutionPolicy { eParallel, eParallelUnsequenced, eSequenced };

    /**
     * TODO: Create .cpp + Move to LIB
     *
     */
    class GraphicsPrimaryCommandBuffer : public CommandBuffer<vk::RenderPass, vk::Framebuffer, vk::Extent2D, int> {
       public:
        using SecondaryCommandBuffer = CommandBuffer<vk::CommandBufferInheritanceInfo const&, vk::Extent2D, int>;

        GraphicsPrimaryCommandBuffer(
            vk::CommandBuffer command_buffer, ExecutionPolicy policy,
            std::function<void(vk::CommandBuffer, vk::ArrayProxy<vk::CommandBuffer const>, vk::RenderPass, vk::Framebuffer, vk::Extent2D, int)>
                update_code,
            std::function<vk::CommandBufferInheritanceInfo(vk::RenderPass, vk::Framebuffer)> inheritance_create)
            : CommandBuffer<vk::RenderPass, vk::Framebuffer, vk::Extent2D, int>(
                  command_buffer, [](vk::CommandBuffer command_buffer, vk::RenderPass render_pass, vk::Framebuffer frame,
                                     vk::Extent2D swapchain_extent, int frame_index) {}),
              inheritance_create(inheritance_create),
              policy(policy) {
            this->update_code = [update_code, &secondary = this->secondary_command_buffers](vk::CommandBuffer command_buffer,
                                                                                            vk::RenderPass render_pass, vk::Framebuffer frame,
                                                                                            vk::Extent2D swapchain_extent, int frame_index) {
                update_code(command_buffer, secondary, render_pass, frame, swapchain_extent, frame_index);
            };
        };

        virtual ~GraphicsPrimaryCommandBuffer() = default;

        /**
         * @brief Add a secondary command buffer
         *
         * @param command_buffer Command buffer
         */
        void addSecondary(SecondaryCommandBuffer* command_buffer) {
            std::lock_guard lock(this->mutex);

            // Add to arrays
            this->secondary_command_buffers.push_back(*command_buffer);
            this->secondary_command_buffers_.push_back(command_buffer);

            // Add to_update
            if (command_buffer->state() == CommandBufferState::eOutdate) {
                this->to_update.push_back(command_buffer);
            }
        }

        virtual CommandBufferState state() const override {
            std::lock_guard lock(this->mutex);

            if (!this->to_update.empty()) {
                return CommandBufferState::eOutdate;
            }
            return this->state_;
        }

        virtual void update(vk::RenderPass render_pass, vk::Framebuffer frame, vk::Extent2D swapchain_extent, int frame_index) override {
            std::lock_guard lock(this->mutex);

            /* SECONDARY CMD PART */

            if (!this->to_update.empty()) {
                // Create inheritance info for the secondary command buffers
                auto inheritance_info = this->inheritance_create(render_pass, frame);

                switch (this->policy) {
                    case ExecutionPolicy::eParallel:
#if defined(__GNUC__) && (__GNUC___ < 9)
                        tbb::parallel_for_each(this->to_update.begin(), this->to_update.end(),
                                               [&](auto command_buffer) { command_buffer->update(inheritance_info, swapchain_extent, frame_index); });
#else
                        std::for_each(std::execution::par, this->to_update.begin(), this->to_update.end(),
                                      [&](auto command_buffer) { command_buffer->update(inheritance_info, swapchain_extent, frame_index); });
#endif
                        break;

                    case ExecutionPolicy::eParallelUnsequenced:
#if defined(__GNUC__) && (__GNUC___ < 9)
                        tbb::parallel_for_each(this->to_update.begin(), this->to_update.end(),
                                               [&](auto command_buffer) { command_buffer->update(inheritance_info, swapchain_extent, frame_index); });
#else
                        std::for_each(std::execution::par_unseq, this->to_update.begin(), this->to_update.end(),
                                      [&](auto command_buffer) { command_buffer->update(inheritance_info, swapchain_extent, frame_index); });
#endif
                        break;

                    case ExecutionPolicy::eSequenced:
#if defined(__GNUC__) && (__GNUC___ < 9)
                        for (auto command_buffer : this->to_update) {
                            command_buffer->update(inheritance_info, swapchain_extent, frame_index);
                        }
#else
                        std::for_each(std::execution::seq, this->to_update.begin(), this->to_update.end(),
                                      [&](auto command_buffer) { command_buffer->update(inheritance_info, swapchain_extent, frame_index); });
#endif
                        break;

                    default:
                        throw ao::core::Exception(fmt::format("Unknown execution policy: {}", static_cast<int>(this->policy)));
                }
                this->to_update.clear();
            }

            /* PRIMARY CMD PART */

            if (this->state_ == CommandBufferState::eOutdate) {
                this->update_code(this->command_buffer, render_pass, frame, swapchain_extent, frame_index);

                this->state_ = CommandBufferState::UpToDate;
            }
        }

        /**
         * @brief Invalidate primary/secondary command buffers
         *
         */
        virtual void invalidate() override {
            std::lock_guard lock(this->mutex);

            this->to_update.reserve(this->to_update.size() + this->secondary_command_buffers_.size());
            for (auto buffer : this->secondary_command_buffers_) {
                this->to_update.push_back(buffer);
            }

            this->state_ = CommandBufferState::eOutdate;
        }

        /**
         * @brief Invalidate secondary command buffers
         *
         */
        virtual void invalidateSecondary() {
            std::lock_guard lock(this->mutex);

            this->to_update.reserve(this->to_update.size() + this->secondary_command_buffers_.size());
            for (auto buffer : this->secondary_command_buffers_) {
                this->to_update.push_back(buffer);
            }
        }

       protected:
        std::vector<SecondaryCommandBuffer*> secondary_command_buffers_;
        std::vector<vk::CommandBuffer> secondary_command_buffers;
        std::vector<SecondaryCommandBuffer*> to_update;

        std::function<vk::CommandBufferInheritanceInfo(vk::RenderPass, vk::Framebuffer)> inheritance_create;
        mutable std::mutex mutex;
        ExecutionPolicy policy;
    };
}  // namespace ao::vulkan