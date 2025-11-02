#pragma once

#include <string>

#include <Base/Cache.h>

#include <Network/Definitions.h>

#include <PersisterInterface/DeserializedSimulation.h>

#include "DownloadCache.h"

struct DownloadNetworkResourceRequestData
{
    std::string resourceId;
    std::string resourceName;
    std::string resourceVersion;
    NetworkResourceType resourceType = NetworkResourceType_Simulation;
    DownloadCache downloadCache;
};
