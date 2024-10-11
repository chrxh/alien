#pragma once

#include <string>

#include "DownloadCache.h"
#include "Base/Cache.h"
#include "EngineInterface/DeserializedSimulation.h"
#include "Network/Definitions.h"

struct DownloadNetworkResourceRequestData
{
    std::string resourceId;
    std::string resourceName;
    std::string resourceVersion;
    NetworkResourceType resourceType = NetworkResourceType_Simulation;
    DownloadCache downloadCache;
};
