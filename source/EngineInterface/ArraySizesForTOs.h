#pragma once

#include <stdint.h>

struct ArraySizesForTOs
{
    uint64_t creatures = 0;
    uint64_t genomes = 0;
    uint64_t genes = 0;
    uint64_t nodes = 0;
    uint64_t objects = 0;
    uint64_t energyParticles = 0;
    uint64_t heap = 0;

    bool operator==(ArraySizesForTOs const& other) const = default;
};
