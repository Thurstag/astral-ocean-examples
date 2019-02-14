// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <string>
#include <vector>

#ifdef __has_include
#    if __has_include(<vld.h>)
#        include <vld.h>
#    endif
#endif

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
        virtual ~GLFWEngine() = default;

        /**
         * @brief Define callback on framebuffer resize
         *
         * @param window Window
         * @param width Window's width
         * @param height Window's height
         */
        static void OnFramebufferSizeCallback(GLFWwindow* window, int width, int height);

        /**
         * @brief Define callback on key event
         *
         * @param window Window
         * @param key Key code
         * @param scancode Scan code
         * @param action Action
         * @param mods Mods
         */
        static void OnKeyEventCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

        /**
         * @brief Define callback on key event
         *
         * @param window Window
         * @param key Key code
         * @param scancode Scan code
         * @param action Action
         * @param mods Mods
         */
        virtual void onKeyEventCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

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

        std::unique_ptr<MetricModule> metrics;
        std::map<u64, std::pair<u64, u64>> key_states;

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
    };
}  // namespace ao::vulkan
