// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#include "metric_module.h"

#include <ao/core/utilities/pointers.h>
#include <boost/algorithm/string/join.hpp>

ao::vulkan::MetricModule::MetricModule(std::weak_ptr<Device> device) : device(device) {
    auto _device = ao::core::shared(this->device);

    // Create query pool
    this->timestamp_query_pool =
        _device->logical.createQueryPool(vk::QueryPoolCreateInfo(vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 128));
    this->triangle_query_pool = _device->logical.createQueryPool(vk::QueryPoolCreateInfo(
        vk::QueryPoolCreateFlags(), vk::QueryType::ePipelineStatistics, 4, vk::QueryPipelineStatisticFlagBits::eClippingInvocations));
}

ao::vulkan::MetricModule::~MetricModule() {
    for (auto& pair : this->metrics) {
        delete pair.second;
    }
    this->metrics.clear();

    ao::core::shared(this->device)->logical.destroyQueryPool(this->timestamp_query_pool);
    ao::core::shared(this->device)->logical.destroyQueryPool(this->triangle_query_pool);
}

ao::vulkan::Metric* ao::vulkan::MetricModule::operator[](std::string name) {
    auto it = std::find_if(this->metrics.begin(), this->metrics.end(),
                           [name](std::pair<std::string, ao::vulkan::Metric*> const& pair) { return name == pair.first; });

    if (it == this->metrics.end()) {
        throw core::Exception(fmt::format("Fail to find metric: {0}", name));
    }
    return this->metrics[name];
}

void ao::vulkan::MetricModule::add(std::string name, ao::vulkan::Metric* metric) {
    this->metrics[name] = metric;
}

void ao::vulkan::MetricModule::reset() {
    for (auto& pair : this->metrics) {
        pair.second->reset();
    }
}

vk::QueryPool ao::vulkan::MetricModule::timestampQueryPool() {
    return this->timestamp_query_pool;
}

vk::QueryPool ao::vulkan::MetricModule::triangleQueryPool() {
    return this->triangle_query_pool;
}

std::string ao::vulkan::MetricModule::str() {
    std::vector<std::string> strings;

    for (auto& pair : this->metrics) {
        strings.push_back(fmt::format("{}: {}", pair.first, pair.second->str()));
    }

    return boost::algorithm::join(strings, " - ");
}
