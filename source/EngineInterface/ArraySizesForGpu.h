#pragma once

#include <stdint.h>

struct ArraySizesForGpu
{
    uint64_t cellArray = 0;
    uint64_t particleArray = 0;
    uint64_t heap = 0;
};
