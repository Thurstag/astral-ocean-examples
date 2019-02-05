// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <vector>

#include <GLFW/glfw3.h>
#include <ao/core/utilities/types.h>

namespace ao::vulkan::utilities {
    /**
     * @brief Monitors
     *
     * @return std::vector<GLFWmonitor*> Monitors
     */
    inline std::vector<GLFWmonitor*> monitors() {
        int count;

        // Get monitors
        GLFWmonitor** monitors = glfwGetMonitors(&count);
        return std::vector<GLFWmonitor*>(monitors, monitors + count);
    }

    /**
     * @brief Extensions
     *
     * @return std::vector<char const*> Extensions
     */
    inline std::vector<char const*> extensions() {
        u32 count;

        // Get extensions
        char const** extensions = glfwGetRequiredInstanceExtensions(&count);
        return std::vector<char const*>(extensions, extensions + count);
    }
}  // namespace ao::vulkan::utilities
