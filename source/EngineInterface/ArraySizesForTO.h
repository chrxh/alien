#pragma once

#include <stdint.h>

struct ArraySizesForTO
{
    uint64_t genomes = 0;
    uint64_t genes = 0;
    uint64_t nodes = 0;
    uint64_t cells = 0;
    uint64_t particles = 0;
    uint64_t heap = 0;

	bool operator==(ArraySizesForTO const& other) const = default;
};
