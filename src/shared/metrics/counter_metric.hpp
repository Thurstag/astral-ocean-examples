// Copyright 2018 Astral-Ocean Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.md file included.

#pragma once

#include <chrono>

#include "metric.h"

namespace ao::vulkan {
	/// <summary>
	/// CounterMetric class
	/// </summary>
	template<class Period, class Type>
	class CounterMetric : public Metric {
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="default">Default value</param>
		CounterMetric(Type default) : count(default), old(default), default(default) {};

		/// <summary>
		/// Destructor
		/// </summary>
		virtual ~CounterMetric() = default;

		/// <summary>
		/// Method to increment count
		/// </summary>
		void increment() {
			this->count++;
		}

		/// <summary>
		/// Method to know if it must be reset
		/// </summary>
		/// <returns>True or False</returns>
		bool hasToBeReset() {
			return std::chrono::duration_cast<Period>(std::chrono::system_clock::now() - this->clock).count() > 0;
		}

		void reset() override {
			if (this->hasToBeReset()) {
				this->old = count;

				this->clock = std::chrono::system_clock::now();
				this->count = this->default;
			}
		}

		std::string toString() override {
			return fmt::format("{0}", this->old);
		}
	protected:
		std::chrono::time_point<std::chrono::system_clock> clock;
		Type default;
		Type count;
		Type old;
	};
}

