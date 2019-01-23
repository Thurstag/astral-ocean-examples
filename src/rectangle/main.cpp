// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include <iostream>

#if defined(_DEBUG) && defined(_WIN32)
#include <vld.h>
#endif

#define GLM_FORCE_RADIANS 1

#include <ao/core/exception/exception.h>
#include <ao/vulkan/engine/settings.h>
#include <ao/core/logger/logger.h>

#include "rectangle.h"

struct Main{};

int main(int argc, char* argv[]) {
	ao::core::Logger::Init();

	// Get LOGGER
	ao::core::Logger LOGGER = ao::core::Logger::GetInstance<Main>();

	// Define settings
	ao::vulkan::EngineSettings settings(
		ao::vulkan::WindowSettings("Rectangle", 1280, 720, true),
		ao::vulkan::CoreSettings(std::thread::hardware_concurrency(), true)
	);
	ao::vulkan::Engine* engine;

	try {
		engine = new RectangleDemo(settings);

		// Run engine
		engine->run();
	} catch (ao::core::Exception & e) {
		LOGGER << ao::core::LogLevel::fatal << e;
	} catch (std::exception& e) {
		LOGGER << ao::core::LogLevel::fatal << ao::core::Exception(e.what(), false);
	} catch (...) {
		LOGGER << ao::core::LogLevel::fatal << "Unknown exception";
	}

	// Free engine
	delete engine;
	return 0;
}
