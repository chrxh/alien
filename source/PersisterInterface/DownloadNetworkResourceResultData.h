#pragma once

#include <string>

#include "PersisterInterface/DeserializedSimulation.h"
#include "EngineInterface/GenomeDescription.h"

struct DownloadNetworkResourceResultData
{
    std::string resourceName;
    std::string resourceVersion;
    NetworkResourceType resourceType = NetworkResourceType_Simulation;
    std::variant<DeserializedSimulation, GenomeDescription> resourceData;
};
