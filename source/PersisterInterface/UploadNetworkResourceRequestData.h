#pragma once

#include <string>

#include <Network/Definitions.h>

#include <EngineInterface/GenomeDesc.h>

#include "DownloadCache.h"

struct UploadNetworkResourceRequestData
{
    std::string folderName;
    std::string resourceWithoutFolderName;
    std::string resourceDescription;
    WorkspaceType workspaceType = WorkspaceType_Public;
    DownloadCache downloadCache;

    struct SimulationData
    {
        float zoom = 1.0f;
        RealVector2D center;
    };
    struct CreatureData
    {
        GenomeDesc description;
    };
    std::variant<SimulationData, CreatureData> data;
};
