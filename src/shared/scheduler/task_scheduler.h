// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <atomic>
#include <functional>
#include <future>
#include <map>
#include <mutex>

#include <ao/core/utilities/types.h>

namespace ao::vulkan {

    class TaskScheduler {
       public:
        /**
         * @brief Construct a new TaskScheduler object
         *
         */
        TaskScheduler();

        /**
         * @brief Destroy the TaskScheduler object
         *
         */
        ~TaskScheduler();

        /**
         * @brief Run scheduler
         *
         */
        void run();

        /**
         * @brief Stop scheduler
         *
         */
        void stop();

        /**
         * @brief Schedule a task
         *
         * @param tick_rate Task's tick rate
         * @param task Task
         */
        void schedule(u16 tick_rate, std::function<void()> task);

       protected:
        std::unique_ptr<std::mutex> tasks_mutex;
        std::atomic_bool running;

        std::map<u16, std::pair<std::mutex, std::vector<std::function<void()>>>> tasks;
        std::vector<std::future<void>> futures;
    };
}  // namespace ao::vulkan