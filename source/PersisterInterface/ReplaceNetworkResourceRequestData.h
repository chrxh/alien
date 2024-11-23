#pragma once

#include <string>

#include "EngineInterface/GenomeDescriptions.h"
#include "Network/Definitions.h"
#include "DownloadCache.h"

struct ReplaceNetworkResourceRequestData
{
    std::string resourceId;
    WorkspaceType workspaceType = WorkspaceType_Public;
    DownloadCache downloadCache;

    struct SimulationData
    {
        float zoom = 1.0f;
        RealVector2D center;
    };
    struct GenomeData
    {
        GenomeDescription description;
    };
    std::variant<SimulationData, GenomeData> data;
};
