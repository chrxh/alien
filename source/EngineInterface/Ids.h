#pragma once

#include <cstdint>

struct Ids
{
    uint64_t entityId = 0;  // unified counter for cells, particles, creatures and genomes
    uint32_t lineageId = 0;
};
