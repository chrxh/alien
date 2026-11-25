#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <Network/Definitions.h>

#include <Data/GenomeDesc.h>

#include "DownloadCache.h"

struct ReplaceNetworkResourceRequestData
{
    std::string resourceId;
    WorkspaceType workspaceType = WorkspaceType_Community;
    bool announceOnDiscord = true;
    DownloadCache downloadCache;

    struct SimulationData
    {
        float zoom = 1.0f;
        RealVector2D center;
        std::string jpg;
        std::optional<std::filesystem::path> filename;  // Replaces by this file instead of the running simulation
    };
    struct CreatureData
    {
        GenomeDesc description;
        std::string jpg;
    };
    std::variant<SimulationData, CreatureData> data;
};
