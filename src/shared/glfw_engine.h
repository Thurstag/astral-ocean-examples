// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <string>
#include <vector>

#include <ao/vulkan/engine/engine.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "metrics/metric_module.h"
#include "utilities/glfw.h"

namespace ao::vulkan {
    namespace settings {
        constexpr char* WindowResizable = "window.resizable";
        constexpr char* WindowTitle = "window.title";
    }  // namespace settings

    class GLFWEngine : public virtual Engine {
       public:
        explicit GLFWEngine(std::shared_ptr<EngineSettings> settings) : Engine(settings), window(nullptr){};
        virtual ~GLFWEngine();

       protected:
        GLFWwindow* window;

        void initWindow() override;
        void initSurface(vk::SurfaceKHR& surface) override;
        void freeWindow() override;
        bool isIconified() const override;

        void freeVulkan() override;
        void initVulkan() override;
        void render() override;
        bool loopingCondition() const override;
        void waitMaximized() override;

        std::vector<char const*> instanceExtensions() const override;
        void updateCommandBuffers() override;

        virtual void afterFrame() override;

       private:
        std::unique_ptr<MetricModule> metrics;
    };
}  // namespace ao::vulkan
