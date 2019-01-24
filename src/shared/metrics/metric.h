// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

namespace ao::vulkan {
    /// <summary>
    /// Metric class
    /// </summary>
    class Metric {
       public:
        /// <summary>
        /// Constructor
        /// </summary>
        Metric() = default;

        /// <summary>
        /// Destructor
        /// </summary>
        virtual ~Metric() = default;

        /// <summary>
        /// Method to reset
        /// </summary>
        virtual void reset() = 0;

        /// <summary>
        /// Method to string
        /// </summary>
        /// <returns>String representation</returns>
        virtual std::string toString() = 0;
    };
}  // namespace ao::vulkan
