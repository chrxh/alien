#pragma once

#include <vector>

#include "GenomeDescriptions.h"


class GenomeEncoder
{
public:
    static std::vector<uint8_t> encode(std::vector<CellGenomeDescription> const& cells);
};
