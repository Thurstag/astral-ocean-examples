// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include "command_buffer.hpp"

namespace ao::vulkan {
    /**
     * @brief Secondary command buffer implementation.
     *
     * Update code parameters:
     *  - vk::Extent2D = swapchain's extent
     *  - int: Frame index
     *
     */
    class SecondaryCommandBuffer : public CommandBuffer<vk::CommandBufferInheritanceInfo const&, vk::Extent2D, int> {
       public:
        /**
         * @brief Construct a new SecondaryCommandBufferobject
         *
         * @param command_buffer Command buffer
         * @param update_code Update code applied to command buffer
         */
        SecondaryCommandBuffer(vk::CommandBuffer command_buffer,
                               std::function<void(vk::CommandBuffer, vk::CommandBufferInheritanceInfo const&, vk::Extent2D, int)> update_code)
            : CommandBuffer<vk::CommandBufferInheritanceInfo const&, vk::Extent2D, int>(command_buffer, update_code){};

        /**
         * @brief Destroy the SecondaryCommandBuffer object
         *
         */
        virtual ~SecondaryCommandBuffer() = default;
    };
}  // namespace ao::vulkan