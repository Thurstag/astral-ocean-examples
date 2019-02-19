// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <string>
#include <type_traits>

#include <ao/core/utilities/pointers.h>
#include <ao/vulkan/utilities/vulkan.h>
#include <ao/vulkan/wrapper/device.h>
#include <vulkan/vulkan.hpp>

#include "metric.h"

namespace ao::vulkan {
    /**
     * @brief Duration metric
     *
     */
    class DurationMetric : public Metric {
       public:
        /**
         * @brief Construct a new DurationMetric object
         *
         * @param unit Unit
         */
        DurationMetric(std::string unit) : unit(unit){};

        /**
         * @brief Destroy the DurationMetric object
         *
         */
        virtual ~DurationMetric() = default;

        /**
         * @brief Start
         *
         */
        virtual void start() = 0;

        /**
         * @brief Stop
         *
         */
        virtual void stop() = 0;

       protected:
        std::string unit;
    };

    /**
     * @brief Basic implementation of DurationMetric
     *
     * @tparam Period
     * @tparam 2
     */
    template<class Period, size_t Precision = 2>
    class BasicDurationMetric : public DurationMetric {
       public:
        /**
         * @brief Construct a new BasicDurationMetric object
         *
         * @param unit Unit
         */
        BasicDurationMetric(std::string unit) : DurationMetric(unit){};

        /**
         * @brief Destroy the BasicDurationMetric object
         *
         */
        virtual ~BasicDurationMetric() = default;

        void start() override {
            this->_start = std::chrono::system_clock::now();
        }

        void stop() override {
            this->_stop = std::chrono::system_clock::now();
        }

        void reset() override {
            this->_start = {};
            this->_stop = {};
        }

        std::string str() override {
            return fmt::format("{:.{}f} {}", std::chrono::duration_cast<Period>(this->_stop - this->_start).count(), Precision, this->unit);
        }

       private:
        std::chrono::time_point<std::chrono::system_clock> _start;
        std::chrono::time_point<std::chrono::system_clock> _stop;
    };

    /**
     * @brief Duration metric for command buffer
     *
     * @tparam Period Period
     * @tparam Index Query index
     * @tparam Precision Precision
     */
    template<class Period, size_t Index = 0, size_t Precision = 4>
    class DurationCommandBufferMetric : public DurationMetric {
       public:
        /**
         * @brief Construct a new DurationCommandBufferMetric object
         *
         * @param unit
         * @param module
         */
        DurationCommandBufferMetric(std::string unit, std::pair<std::weak_ptr<ao::vulkan::Device>, vk::QueryPool> module)
            : DurationMetric(unit), module(module) {
            this->period = ao::core::shared(this->module.first)->physical.getProperties().limits.timestampPeriod;
        };

        /**
         * @brief Destroy the DurationCommandBufferMetric object
         *
         */
        virtual ~DurationCommandBufferMetric() = default;

        void start() override {}

        void stop() override {}

        void reset() override {}

        std::string str() override {
            std::array<u64, 2> results;

            // Get results
            ao::vulkan::utilities::vkAssert(
                ao::core::shared(this->module.first)
                    ->logical.getQueryPoolResults(this->module.second, Index, 2, results.size() * sizeof(u64), results.data(), sizeof(u64),
                                                  vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait),
                "Fail to get query results");

            return fmt::format("{:.{}f} {}", (static_cast<double>(results[1]) - results[0]) * this->period / (std::nano::den / Period::den),
                               Precision, this->unit);
        }

       private:
        std::pair<std::weak_ptr<ao::vulkan::Device>, vk::QueryPool> module;
        float period;
    };
}  // namespace ao::vulkan
