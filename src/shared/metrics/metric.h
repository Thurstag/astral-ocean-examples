// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

namespace ao::vulkan {
    /**
     * @brief Metric
     *
     */
    class Metric {
       public:
        /**
         * @brief Construct a new Metric object
         *
         */
        Metric() = default;

        /**
         * @brief Destroy the Metric object
         *
         */
        virtual ~Metric() = default;

        /**
         * @brief Reset metrics
         *
         */
        virtual void reset() = 0;

        /**
         * @brief String representation
         *
         * @return std::string String
         */
        virtual std::string str() = 0;
    };
}  // namespace ao::vulkan
