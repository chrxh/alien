#pragma once

#include <vector>

#include "GenomeDescriptions.h"


class GenomeEncoder
{
public:
    static std::vector<uint8_t> encode(GenomeDescription const& cells);
};
