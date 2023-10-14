#pragma once

#include <vector>

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"

struct GenomeEncodingSpecification
{
    MEMBER_DECLARATION(GenomeEncodingSpecification, bool, numRepetitions, true);
    MEMBER_DECLARATION(GenomeEncodingSpecification, bool, concatenationAngle1, true);
    MEMBER_DECLARATION(GenomeEncodingSpecification, bool, concatenationAngle2, true);
};

class GenomeDescriptionConverter
{
public:
    static std::vector<uint8_t> convertDescriptionToBytes(GenomeDescription const& genome, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());
    static GenomeDescription convertBytesToDescription(std::vector<uint8_t> const& data, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());

    static int convertNodeAddressToNodeIndex(std::vector<uint8_t> const& data, int nodeAddress, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());
    static int convertNodeIndexToNodeAddress(std::vector<uint8_t> const& data, int nodeIndex, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());
    static int getNumNodesRecursively(std::vector<uint8_t> const& data, bool includeRepetitions, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());
};
