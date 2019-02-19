// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <chrono>

#include "metric.h"

namespace ao::vulkan {
    /**
     * @brief Counter metric
     *
     * @tparam Period Period
     * @tparam Type Counter type
     */
    template<class Period, class Type>
    class CounterMetric : public Metric {
       public:
        /**
         * @brief Construct a new CounterMetric object
         *
         * @param default Default value
         */
        CounterMetric(Type default) : count(default), old(default), default(default){};

        /**
         * @brief Destroy the CounterMetric object
         *
         */
        virtual ~CounterMetric() = default;

        /**
         * @brief Increment counter
         *
         * @param count Count
         */
        void increment(Type count = 1) {
            this->count += count;
        }

        /**
         * @brief
         *
         * @return true Counter need to be reset
         * @return false Counter doesn't need to be reset
         */
        bool hasToBeReset() {
            return std::chrono::duration_cast<Period>(std::chrono::system_clock::now() - this->clock).count() > 0;
        }

        void reset() override {
            if (this->hasToBeReset()) {
                this->old = count;

                this->clock = std::chrono::system_clock::now();
                this->count = this->default;
            }
        }

        std::string str() override {
            return fmt::format("{0}", this->old);
        }

       protected:
        std::chrono::time_point<std::chrono::system_clock> clock;
        Type default;
        Type count;
        Type old;
    };

    /**
     * @brief Counter metric for command buffer
     *
     * @tparam Period
     * @tparam Type
     * @tparam Index
     */
    template<class Period, class Type, size_t Index = 0>
    class CounterCommandBufferMetric : public CounterMetric<Period, Type> {
       public:
        CounterCommandBufferMetric(Type default, std::pair<std::weak_ptr<ao::vulkan::Device>, vk::QueryPool> module)
            : CounterMetric(default), module(module) {}

        void increment() = delete;

        void update() {
            u64 result;

            // Get results
            ao::vulkan::utilities::vkAssert(ao::core::shared(this->module.first)
                                                ->logical.getQueryPoolResults(this->module.second, Index, 1, sizeof(u64), &result, sizeof(u64),
                                                                              vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait),
                                            "Fail to get query results");

            this->count += result;
        }

        std::string str() override {
            if (this->old < 1000) {
                return fmt::format("{0}", this->old);
            }

            auto suffixes = "KMGTPE";
            double exp = std::log10(this->old) / std::log10(1000);

            return fmt::format("{:.{}f}{}", this->old / std::pow(1000, static_cast<int>(exp)), 2, suffixes[static_cast<int>(exp) - 1]);
        }

       private:
        std::pair<std::weak_ptr<ao::vulkan::Device>, vk::QueryPool> module;
    };
}  // namespace ao::vulkan
