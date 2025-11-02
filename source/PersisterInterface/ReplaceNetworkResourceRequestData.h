#pragma once

#include <string>

#include <Network/Definitions.h>

#include <EngineInterface/GenomeDescription.h>

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
    struct CreatureData
    {
        GenomeDescription description;
    };
    std::variant<SimulationData, CreatureData> data;
};
