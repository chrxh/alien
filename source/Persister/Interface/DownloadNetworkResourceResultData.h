#pragma once

#include <string>

#include <Network/Interface/Definitions.h>

#include <Data/Interface/Descs.h>
#include <Data/Interface/GenomeDesc.h>

struct DownloadNetworkResourceResultData
{
    std::string resourceName;
    std::string resourceVersion;
    NetworkResourceType resourceType = NetworkResourceType_Simulation;
    std::variant<SimulationDesc, GenomeDesc> resourceData;
};
