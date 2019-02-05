// Copyright 2018 Astral-Ocean Project
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
}  // namespace ao::vulkan
