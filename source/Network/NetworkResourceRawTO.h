#pragma once

#include <string>
#include <map>

#include "Definitions.h"

struct ImGuiTableColumnSortSpecs;

enum NetworkResourceColumnId
{
    NetworkResourceColumnId_Timestamp,
    NetworkResourceColumnId_UserName,
    NetworkResourceColumnId_SimulationName,
    NetworkResourceColumnId_Description,
    NetworkResourceColumnId_Likes,
    NetworkResourceColumnId_NumDownloads,
    NetworkResourceColumnId_Width,
    NetworkResourceColumnId_Height,
    NetworkResourceColumnId_Particles,
    NetworkResourceColumnId_FileSize,
    NetworkResourceColumnId_Version,
    NetworkResourceColumnId_Actions
};

struct _NetworkResourceRawTO
{
    std::string id;
    std::string timestamp;
    std::string userName;
    std::string resourceName;
    std::map<int, int> numLikesByEmojiType;
    int numDownloads;
    int width;
    int height;
    int particles;
    uint64_t contentSize;
    std::string description;
    std::string version;
    int workspaceType;
    NetworkResourceType resourceType;

    static int compare(NetworkResourceRawTO const& left, NetworkResourceRawTO const& right, std::vector<ImGuiTableColumnSortSpecs> const& sortSpecs);
    bool matchWithFilter(std::string const& filter) const;

    int getTotalLikes() const;
};
