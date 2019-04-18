// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <functional>

#include <vulkan/vulkan.hpp>

namespace ao::vulkan {

    /**
     * @brief Command buffer's states
     *
     */
    enum class CommandBufferState { eOutdate, UpToDate };

    /**
     * @brief vk::CommandBuffer wrapper
     *
     * @tparam Update code parameters
     */
    template<class... T>
    class CommandBuffer {
       public:
        /**
         * @brief Construct a new CommandBuffer object
         *
         * @param command_buffer Command buffer
         * @param update_code Update code applied to command buffer
         */
        CommandBuffer(vk::CommandBuffer command_buffer, std::function<void(vk::CommandBuffer, T...)> update_code)
            : command_buffer(command_buffer), update_code(update_code), state_(CommandBufferState::eOutdate){};

        /**
         * @brief Destroy the CommandBuffer object
         *
         */
        ~CommandBuffer() = default;

        /**
         * @brief Get state
         *
         * @return CommandBufferState State
         */
        virtual CommandBufferState state() const {
            return this->state_;
        }

        /**
         * @brief Conversion into vk::CommandBuffer
         *
         * @return vk::CommandBuffer Command buffer
         */
        operator vk::CommandBuffer() const {
            return this->command_buffer;
        }

        /**
         * @brief Update command buffer
         *
         * @param params Parameters
         */
        virtual void update(T... params) {
            this->update_code(this->command_buffer, params...);

            this->state_ = CommandBufferState::UpToDate;
        }

        /**
         * @brief Invalidate command buffer
         *
         */
        virtual void invalidate() {
            this->state_ = CommandBufferState::eOutdate;
        }

       protected:
        std::function<void(vk::CommandBuffer, T...)> update_code;
        vk::CommandBuffer command_buffer;
        CommandBufferState state_;
    };
}  // namespace ao::vulkan