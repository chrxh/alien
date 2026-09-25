#pragma once

#include <optional>
#include <string>

#include <Base/Definitions.h>

struct ParametersFilter
{
    auto operator<=>(ParametersFilter const&) const = default;

    std::optional<std::string> containedText;
};
