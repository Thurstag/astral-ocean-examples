// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <string>
#include <vector>

#include <ao/vulkan/engine/ao_engine.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "metrics/metric_module.h"
#include "utilities/glfw.h"

namespace ao::vulkan {
	class GLFWEngine : public virtual AOEngine {
	public:
		explicit GLFWEngine(EngineSettings settings) : AOEngine(settings), window(nullptr) {};
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
	private:
		std::unique_ptr<MetricModule> metrics;
	};
}
