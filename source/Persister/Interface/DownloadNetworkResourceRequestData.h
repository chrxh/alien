#pragma once

#include <string>

#include <Base/Interface/Cache.h>

#include <Network/Interface/Definitions.h>

#include "DownloadCache.h"

struct DownloadNetworkResourceRequestData
{
    std::string resourceId;
    std::string resourceName;
    std::string resourceVersion;
    NetworkResourceType resourceType = NetworkResourceType_Simulation;
    DownloadCache downloadCache;
};
