#pragma once

#include <vector>

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"


class GenomeDescriptionConverter
{
public:
    static std::vector<uint8_t> convertDescriptionToBytes(GenomeDescription const& genome);
    static GenomeDescription convertBytesToDescription(std::vector<uint8_t> const& data, SimulationParameters const& parameters);
};
