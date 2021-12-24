#pragma once

#include <stdint.h>

namespace Const
{
    auto constexpr MaxInspectedEntities = 20;
}

struct InspectedEntityIds
{
    uint64_t values[Const::MaxInspectedEntities];
};