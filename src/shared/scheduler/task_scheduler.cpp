// Copyright 2018-2019 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "task_scheduler.h"

#if defined(__GNUC__) && (__GNUC___ < 9)
#    include "tbb/parallel_for_each.h"
#else
#    include <execution>
#endif

ao::vulkan::TaskScheduler::TaskScheduler() : running(false), tasks_mutex(std::make_unique<std::mutex>()) {}

ao::vulkan::TaskScheduler::~TaskScheduler() {
    std::lock_guard lock(*this->tasks_mutex);

    if (this->running) {
        this->stop();
    }
}

void ao::vulkan::TaskScheduler::run() {
    std::lock_guard lock(*this->tasks_mutex);
    this->running = true;

    // Run tasks
    for (auto it = this->tasks.begin(); it != this->tasks.end(); it++) {
        this->futures.push_back(
            std::async(std::launch::async, [rate = it->first, &mutex = it->second.first, &tasks = it->second.second, &running = this->running]() {
                while (running) {
                    mutex.lock();
                    {
#if defined(__GNUC__) && (__GNUC___ < 9)
                        tbb::parallel_for_each(tasks.begin(), tasks.end(), [](auto task) { task(); });
#else
                        std::for_each(std::execution::par_unseq, tasks.begin(), tasks.end(), [](auto task) { task(); });
#endif
                    }
                    mutex.unlock();

                    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / rate));
                }
            }));
    }
}

void ao::vulkan::TaskScheduler::stop() {
    std::lock_guard lock(*this->tasks_mutex);
    this->running = false;

    // Wait all tasks
    for (auto& future : this->futures) {
        future.wait();
    }
}

void ao::vulkan::TaskScheduler::schedule(u16 tick_rate, std::function<void()> task) {
    std::lock_guard lock(*this->tasks_mutex);

    // Add task
    if (this->running) {
        if (this->tasks.count(tick_rate) == 0) {  // New tick rate
            this->tasks[tick_rate].second.push_back(task);

            this->futures.push_back(std::async(std::launch::async, [tick_rate, &mutex = this->tasks[tick_rate].first,
                                                                    &tasks = this->tasks[tick_rate].second, &running = this->running]() {
                while (running) {
                    mutex.lock();
                    {
#if defined(__GNUC__) && (__GNUC___ < 9)
                        tbb::parallel_for_each(tasks.begin(), tasks.end(), [](auto task) { task(); });
#else
                        std::for_each(std::execution::par_unseq, tasks.begin(), tasks.end(), [](auto task) { task(); });
#endif
                    }
                    mutex.unlock();

                    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / tick_rate));
                }
            }));
        } else {
            std::lock_guard lock_2(this->tasks[tick_rate].first);

            this->tasks[tick_rate].second.push_back(task);
        }
    } else {
        this->tasks[tick_rate].second.push_back(task);
    }
}