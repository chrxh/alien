#pragma once

#include <vector>

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"


class GenomeTranslator
{
public:
    static std::vector<uint8_t> encode(GenomeDescription const& cells);
    static GenomeDescription decode(std::vector<uint8_t> const& data, SimulationParameters const& parameters);
};
