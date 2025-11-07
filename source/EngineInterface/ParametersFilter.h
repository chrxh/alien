#pragma once

#include <optional>
#include <string>

#include <Base/Definitions.h>

struct ParametersFilter
{
    std::optional<std::string> containedText;
};

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
