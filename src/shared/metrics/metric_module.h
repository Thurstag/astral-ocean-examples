// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <map>

#include <ao/core/exception/exception.h>
#include <ao/vulkan/engine/wrapper/device.h>
#include <vulkan/vulkan.hpp>

#include "metric.h"

namespace ao::vulkan {
    /**
     * @brief Metric module
     *
     */
    class MetricModule {
       public:
        /**
         * @brief Construct a new MetricModule object
         *
         * @param device Device
         */
        explicit MetricModule(std::weak_ptr<Device> device);

        /**
         * @brief Destroy the MetricModule object
         *
         */
        virtual ~MetricModule();

        /**
         * @brief Operator[]
         *
         * @param name Metric's name
         * @return Metric* Metric
         */
        Metric* operator[](std::string name);

        /**
         * @brief Add a metric
         *
         * @param name Metric's name
         * @param metric Metric
         */
        void add(std::string name, Metric* metric);

        /**
         * @brief Reset
         *
         */
        void reset();

        /**
         * @brief Query pool
         *
         * @return vk::QueryPool Query pool
         */
        vk::QueryPool queryPool();

        /**
         * @brief String representation
         *
         * @return std::string String
         */
        std::string str();

       private:
        std::unordered_map<std::string, Metric*> metrics;

        vk::QueryPool _queryPool;
        std::weak_ptr<Device> device;
    };
}  // namespace ao::vulkan
