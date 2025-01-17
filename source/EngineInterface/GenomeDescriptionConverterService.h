#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"

struct GenomeEncodingSpecification
{
    MEMBER_DECLARATION(GenomeEncodingSpecification, bool, numRepetitions, true);
    MEMBER_DECLARATION(GenomeEncodingSpecification, bool, concatenationAngle1, true);
    MEMBER_DECLARATION(GenomeEncodingSpecification, bool, concatenationAngle2, true);
};

class GenomeDescriptionConverterService
{
    MAKE_SINGLETON(GenomeDescriptionConverterService);
public:
    std::vector<uint8_t> convertDescriptionToBytes(GenomeDescription const& genome, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());
    GenomeDescription convertBytesToDescription(std::vector<uint8_t> const& data, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());

    int convertNodeAddressToNodeIndex(std::vector<uint8_t> const& data, int nodeAddress, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());
    int convertNodeIndexToNodeAddress(std::vector<uint8_t> const& data, int nodeIndex, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());
    int getNumNodesRecursively(std::vector<uint8_t> const& data, bool includeRepetitions, GenomeEncodingSpecification const& spec = GenomeEncodingSpecification());
    int getNumRepetitions(std::vector<uint8_t> const& data);
};
