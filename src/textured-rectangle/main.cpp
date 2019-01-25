// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include <iostream>

#if defined(_DEBUG) && defined(_WIN32)
#include <vld.h>
#endif

#define GLM_FORCE_RADIANS 1

#include <ao/core/exception/exception.h>
#include <ao/core/logger/logger.h>
#include <ao/vulkan/engine/settings.h>

#include "textured_rectangle.h"

struct Main {};

int main(int argc, char* argv[]) {
    ao::core::Logger::Init();

    // Get LOGGER
    ao::core::Logger LOGGER = ao::core::Logger::GetInstance<Main>();

    // Define settings
    std::shared_ptr<ao::vulkan::EngineSettings> settings = std::make_shared<ao::vulkan::EngineSettings>();
    settings->get<std::string>(ao::vulkan::settings::WindowTitle) = std::string("Textured Rectangle");
    settings->get<u64>(ao::vulkan::settings::WindowWidth) = 1280;
    settings->get<u64>(ao::vulkan::settings::WindowHeight) = 720;
    settings->get<bool>(ao::vulkan::settings::ValidationLayers) = true;

    ao::vulkan::Engine* engine;
    try {
        engine = new TexturedRectangle(settings);

        // Run engine
        engine->run();
    } catch (ao::core::Exception& e) {
        LOGGER << ao::core::Logger::Level::fatal << e;
    } catch (std::exception& e) {
        LOGGER << ao::core::Logger::Level::fatal << ao::core::Exception(e.what(), false);
    } catch (...) {
        LOGGER << ao::core::Logger::Level::fatal << "Unknown exception";
    }

    // Free engine
    delete engine;
    return 0;
}
