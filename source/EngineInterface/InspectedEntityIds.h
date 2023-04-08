#pragma once

#include <stdint.h>

namespace Const
{
    auto constexpr MaxInspectedObjects = 20;
}

struct InspectedEntityIds
{
    uint64_t values[Const::MaxInspectedObjects];
};