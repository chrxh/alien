#pragma once

#include <string>
#include <map>

#include "Definitions.h"

class ImGuiTableSortSpecs;

enum NetworkDataColumnId
{
    NetworkDataColumnId_Timestamp,
    NetworkDataColumnId_UserName,
    NetworkDataColumnId_SimulationName,
    NetworkDataColumnId_Description,
    NetworkDataColumnId_Likes,
    NetworkDataColumnId_NumDownloads,
    NetworkDataColumnId_Width,
    NetworkDataColumnId_Height,
    NetworkDataColumnId_Particles,
    NetworkDataColumnId_FileSize,
    NetworkDataColumnId_Version,
    NetworkDataColumnId_Actions
};

using NetworkDataType = int;
enum NetworkDataType_
{
    NetworkDataType_Simulation,
    NetworkDataType_Genome
};

struct _NetworkDataTO
{
    std::string id;
    std::string timestamp;
    std::string userName;
    std::string simName;
    std::map<int, int> numLikesByEmojiType;
    int numDownloads;
    int width;
    int height;
    int particles;
    uint64_t contentSize;
    std::string description;
    std::string version;
    bool fromRelease;
    NetworkDataType type;

    static int compare(NetworkDataTO const& left, NetworkDataTO const& right, ImGuiTableSortSpecs const* specs);
    bool matchWithFilter(std::string const& filter) const;

    int getTotalLikes() const;
};
