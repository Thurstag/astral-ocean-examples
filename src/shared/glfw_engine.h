// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <string>
#include <vector>

#if defined(_DEBUG) && defined(_WIN32)
#    ifdef __has_include
#        if __has_include(<vld.h>)
#            include <vld.h>
#        endif
#    endif
#endif

#include <ao/vulkan/engine/engine.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "command_buffer/graphics_primary_command_buffer.h"
#include "metrics/metric_module.h"
#include "utilities/glfw.h"

namespace ao::vulkan {
    namespace settings {
        static constexpr char* WindowResizable = "window.resizable";
        static constexpr char* WindowTitle = "window.title";
    }  // namespace settings

    class GLFWEngine : public Engine {
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
        std::vector<GraphicsPrimaryCommandBuffer*> primary_command_buffers;
        std::unique_ptr<CommandPool> secondary_command_pool;
        GLFWwindow* window;

        std::map<u64, std::pair<u64, u64>> key_states;
        std::unique_ptr<MetricModule> metrics;

        /**
         * @brief Create secondary command buffers
         *
         */
        virtual void createSecondaryCommandBuffers() = 0;

        /**
         * @brief Create pipelines
         *
         */
        virtual void createPipelines() = 0;

        /**
         * @brief Create vulkan buffers
         *
         */
        virtual void createVulkanBuffers() = 0;

        void initWindow() override;
        vk::SurfaceKHR createSurface() override;
        void freeWindow() override;
        bool isIconified() const override;

        void freeVulkan() override;
        virtual void initVulkan() override;
        virtual void prepareVulkan() override;
        void createVulkanObjects() override;
        virtual void render() override;
        bool loopingCondition() const override;
        void waitMaximized() override;

        std::vector<vk::PhysicalDeviceFeatures> deviceFeatures() const override;
        std::vector<char const*> instanceExtensions() const override;
        virtual void updateCommandBuffers() override;

        virtual void afterFrame() override;

        virtual std::vector<ao::vulkan::QueueRequest> requestQueues() const override;
    };
}  // namespace ao::vulkan
