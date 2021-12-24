#pragma once

#include <stdint.h>

namespace Const
{
    auto constexpr MaxInspectedEntities = 20;
}

struct EntityIds
{
    uint64_t values[Const::MaxInspectedEntities];
};