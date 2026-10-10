#pragma once

#include <functional>

#include "ParametersFilter.h"

template <>
struct std::hash<ParametersFilter>
{
    size_t operator()(const ParametersFilter& filter) const noexcept
    {
        if (filter.containedText.has_value()) {
            return std::hash<std::string>{}(*filter.containedText);
        }
        return 0;
    }
};
