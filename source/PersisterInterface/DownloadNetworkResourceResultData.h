#pragma once

#include <string>

#include <Network/Definitions.h>

#include <Data/Descs.h>
#include <Data/GenomeDesc.h>

struct DownloadNetworkResourceResultData
{
    std::string resourceName;
    std::string resourceVersion;
    NetworkResourceType resourceType = NetworkResourceType_Simulation;
    std::variant<SimulationDesc, GenomeDesc> resourceData;
};
