#pragma once

#include <string>

#include "EngineInterface/GenomeDescriptions.h"
#include "Network/Definitions.h"

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
    struct GenomeData
    {
        GenomeDescription description;
    };
    std::variant<SimulationData, GenomeData> data;
};
