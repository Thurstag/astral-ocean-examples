// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <string>
#include <vector>

#include <ao/vulkan/engine.h>
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
        explicit GLFWEngine(std::shared_ptr<EngineSettings> settings);
        virtual ~GLFWEngine();

        /**
         * @brief Define callback on framebuffer resize
         *
         * @param window Window
         * @param width Window's width
         * @param height Window's height
         */
        static void OnFramebufferSizeCallback(GLFWwindow* window, int width, int height);

        /**
         * @brief Load pipeline cache
         *
         * @param file File
         * @return std::vector<u8> Data
         */
        static std::vector<u8> LoadCache(std::string const& file);

        /**
         * @brief Save pipeline cache
         *
         * @param directory Diretcory
         * @param filename File's name
         * @param cache Cache
         */
        void saveCache(std::string const& directory, std::string const& filename, vk::PipelineCache cache);

       protected:
        std::unique_ptr<CommandPool> secondary_command_pool;
        std::vector<vk::CommandBuffer> command_buffers;
        GLFWwindow* window;

        void initWindow() override;
        vk::SurfaceKHR createSurface() override;
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

        virtual std::vector<ao::vulkan::QueueRequest> requestQueues() const override;

       private:
        std::unique_ptr<MetricModule> metrics;
    };
}  // namespace ao::vulkan
