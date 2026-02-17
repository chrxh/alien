#pragma once

#include <string>

#include <Network/Definitions.h>

#include <EngineInterface/GenomeDesc.h>

#include <PersisterInterface/DeserializedSimulation.h>

struct DownloadNetworkResourceResultData
{
    std::string resourceName;
    std::string resourceVersion;
    NetworkResourceType resourceType = NetworkResourceType_Simulation;
    std::variant<DeserializedSimulation, GenomeDesc> resourceData;
};
