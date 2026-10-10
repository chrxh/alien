#pragma once

#include <stdint.h>

namespace Const
{
    auto constexpr MaxInspectedObjects = 20;
    uint64_t constexpr MaxInspectedObjects_Break = 0xffffffffffffffff;
}

struct InspectedEntityIds
{
    uint64_t values[Const::MaxInspectedObjects];
};