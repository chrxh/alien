#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"

class GenomeDescriptionConverterService
{
    MAKE_SINGLETON(GenomeDescriptionConverterService);
public:
    std::vector<uint8_t> convertDescriptionToBytes(GenomeDescription const& genome);
    GenomeDescription convertBytesToDescription(std::vector<uint8_t> const& data);

    int convertNodeAddressToNodeIndex(std::vector<uint8_t> const& data, int nodeAddress);
    int convertNodeIndexToNodeAddress(std::vector<uint8_t> const& data, int nodeIndex);
    int getNumNodesRecursively(std::vector<uint8_t> const& data, bool includeRepetitions);
    int getNumRepetitions(std::vector<uint8_t> const& data);
};
