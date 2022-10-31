#pragma once

#include <vector>

#include "GenomeDescriptions.h"


class GenomeTranslator
{
public:
    static std::vector<uint8_t> encode(GenomeDescription const& cells);
};
