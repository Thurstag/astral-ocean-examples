// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <map>

#include <ao/vulkan/engine/wrappers/device.h>
#include <ao/core/exception/exception.h>
#include <vulkan/vulkan.hpp>

#include "metric.h"

namespace ao::vulkan {
	/// <summary>
	/// Metric module class
	/// </summary>
	class MetricModule {
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="device">Device</param>
		explicit MetricModule(std::weak_ptr<Device> device);

		/// <summary>
		/// Destructor
		/// </summary>
		virtual ~MetricModule();

		/// <summary>
		/// Operator[]
		/// </summary>
		/// <param name="name">Metric's name</param>
		Metric* operator[](std::string name);

		/// <summary>
		/// Method to add a metric
		/// </summary>
		/// <param name="name">Metric's ame</param>
		/// <param name="metric">Metric</param>
		void add(std::string name, Metric* metric);

		/// <summary>
		/// Method to reset all metrics
		/// </summary>
		void reset();

		/// <summary>
		/// Method to get query pool
		/// </summary>
		/// <returns></returns>
		vk::QueryPool queryPool();

		/// <summary>
		/// Method to string
		/// </summary>
		/// <returns>String representation</returns>
		std::string toString();
	private:
		std::unordered_map<std::string, Metric*> metrics;

		vk::QueryPool _queryPool;
		std::weak_ptr<Device> device;
	};
}

