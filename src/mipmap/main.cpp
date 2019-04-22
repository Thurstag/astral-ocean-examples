// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include <iostream>

#define GLM_FORCE_RADIANS 1
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 1

#include <ao/core/exception/exception.h>
#include <ao/core/logger/logger.h>
#include <ao/vulkan/engine/settings.h>

#include "mipmap.h"

struct Main {};

int main(int argc, char* argv[]) {
    ao::core::Logger::Init();

    // Get LOGGER
    ao::core::Logger LOGGER = ao::core::Logger::GetInstance<Main>();

    // Define settings
    std::shared_ptr<ao::vulkan::EngineSettings> settings = std::make_shared<ao::vulkan::EngineSettings>();
    settings->get<std::string>(ao::vulkan::settings::WindowTitle) = std::string("Mipmap Texture");
    settings->get<u32>(ao::vulkan::settings::SurfaceWidth) = 1280;
    settings->get<u32>(ao::vulkan::settings::SurfaceHeight) = 720;
    settings->get<bool>(ao::vulkan::settings::ValidationLayers) = true;
    settings->get<bool>(ao::vulkan::settings::StencilBuffer) = true;

    ao::vulkan::Engine* engine;
    bool exception_thrown = false;
    try {
        engine = new MipmapDemo(settings);

        // Run engine
        engine->run();
    } catch (ao::core::Exception& e) {
        LOGGER << ao::core::Logger::Level::fatal << e;
        exception_thrown = true;
    } catch (std::exception& e) {
        LOGGER << ao::core::Logger::Level::fatal << ao::core::Exception(e.what(), false);
        exception_thrown = true;
    } catch (...) {
        LOGGER << ao::core::Logger::Level::fatal << "Unknown exception";
        exception_thrown = true;
    }

    if (exception_thrown) {
        engine->freeVulkan();

        std::cout << "Press enter to continue";
        std::cin.ignore();
    }

    // Free engine
    delete engine;
    return 0;
}
