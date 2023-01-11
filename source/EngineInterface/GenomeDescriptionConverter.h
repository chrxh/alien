#pragma once

#include <vector>

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"


class GenomeDescriptionConverter
{
public:
    static std::vector<uint8_t> convertDescriptionToBytes(GenomeDescription const& genome);
    static GenomeDescription convertBytesToDescription(std::vector<uint8_t> const& data, SimulationParameters const& parameters);

    static int convertByteIndexToCellIndex(std::vector<uint8_t> const& data, int byteIndex);
    static int convertCellIndexToByteIndex(std::vector<uint8_t> const& data, int cellIndex);
};
