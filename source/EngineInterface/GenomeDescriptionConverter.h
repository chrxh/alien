#pragma once

#include <vector>

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"


class GenomeDescriptionConverter
{
public:
    static std::vector<uint8_t> convertDescriptionToBytes(GenomeDescription const& genome);
    static GenomeDescription convertBytesToDescription(std::vector<uint8_t> const& data);

    static int convertNodeAddressToNodeIndex(std::vector<uint8_t> const& data, int nodeAddress);
    static int convertNodeIndexToNodeAddress(std::vector<uint8_t> const& data, int nodeIndex);
    static int getNumNodesRecursively(std::vector<uint8_t> const& data);
};
