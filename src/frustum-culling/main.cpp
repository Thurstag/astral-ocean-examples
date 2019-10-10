// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include <iostream>

#define GLM_FORCE_RADIANS 1

#include <ao/core/exception/exception.h>
#include <ao/core/logging/log.h>
#include <ao/vulkan/engine/settings.h>

#include "frustum_culling.h"

struct Main {};

int main(int argc, char* argv[]) {
    // Init logger
    ao::core::Logger::Init();

    // Define settings
    std::shared_ptr<ao::vulkan::EngineSettings> settings = std::make_shared<ao::vulkan::EngineSettings>();
    settings->get<std::string>(ao::vulkan::settings::WindowTitle) = std::string("Frustum culling");
    settings->get<u32>(ao::vulkan::settings::SurfaceWidth) = 1280;
    settings->get<u32>(ao::vulkan::settings::SurfaceHeight) = 720;
    settings->get<bool>(ao::vulkan::settings::ValidationLayers) = true;
    settings->get<bool>(ao::vulkan::settings::StencilBuffer) = true;

    ao::vulkan::Engine* engine;
    bool exception_thrown = false;
    try {
        engine = new FrustumDemo(settings);

        // Run engine
        engine->run();
    } catch (ao::core::Exception& e) {
        LOG_MSG(fatal) << e;
        exception_thrown = true;
    } catch (std::exception& e) {
        LOG_MSG(fatal) << ao::core::Exception(e.what(), false);
        exception_thrown = true;
    } catch (...) {
        LOG_MSG(fatal) << "Unknown exception";
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
