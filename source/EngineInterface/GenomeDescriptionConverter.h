#pragma once

#include <vector>

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"


class GenomeDescriptionConverter
{
public:
    struct EncodingSpecification
    {
        MEMBER_DECLARATION(EncodingSpecification, bool, numRepetitions, true);
        MEMBER_DECLARATION(EncodingSpecification, bool, concatenationAngle1, true);
        MEMBER_DECLARATION(EncodingSpecification, bool, concatenationAngle2, true);
    };

    static std::vector<uint8_t> convertDescriptionToBytes(GenomeDescription const& genome, EncodingSpecification const& spec = EncodingSpecification());
    static GenomeDescription convertBytesToDescription(std::vector<uint8_t> const& data, EncodingSpecification const& spec = EncodingSpecification());

    static int convertNodeAddressToNodeIndex(std::vector<uint8_t> const& data, int nodeAddress, EncodingSpecification const& spec = EncodingSpecification());
    static int convertNodeIndexToNodeAddress(std::vector<uint8_t> const& data, int nodeIndex, EncodingSpecification const& spec = EncodingSpecification());
    static int getNumNodesRecursively(std::vector<uint8_t> const& data, bool includeRepetitions, EncodingSpecification const& spec = EncodingSpecification());
};
