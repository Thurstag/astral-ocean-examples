// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <functional>

#include "metric.h"

namespace ao::vulkan {
    /**
     * @brief Function based metric
     *
     */
    class LambdaMetric : public Metric {
       public:
        /**
         * @brief Construct a new LambdaMetric object
         *
         * @param function Function
         */
        LambdaMetric(std::function<std::string()> function) : function(function) {}

        /**
         * @brief Destroy the LambdaMetric object
         *
         */
        virtual ~LambdaMetric() = default;

        virtual void reset() override {}

        virtual std::string str() override {
            return this->function();
        }

       protected:
        std::function<std::string()> function;
    };
}  // namespace ao::vulkan