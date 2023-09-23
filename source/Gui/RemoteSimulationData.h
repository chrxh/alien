#pragma once

#include <string>
#include <map>

class ImGuiTableSortSpecs;

enum RemoteSimulationDataColumnId
{
    RemoteSimulationDataColumnId_Timestamp,
    RemoteSimulationDataColumnId_UserName,
    RemoteSimulationDataColumnId_SimulationName,
    RemoteSimulationDataColumnId_Description,
    RemoteSimulationDataColumnId_Likes,
    RemoteSimulationDataColumnId_NumDownloads,
    RemoteSimulationDataColumnId_Width,
    RemoteSimulationDataColumnId_Height,
    RemoteSimulationDataColumnId_Particles,
    RemoteSimulationDataColumnId_FileSize,
    RemoteSimulationDataColumnId_Version,
    RemoteSimulationDataColumnId_Actions
};

using RemoteDataType = int;
enum RemoteDataType_
{
    RemoteDataType_Simulation,
    RemoteDataType_Genome
};

class RemoteSimulationData
{
public:
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
    RemoteDataType type;

    static int compare(void const* left, void const* right, ImGuiTableSortSpecs const* specs);
    bool matchWithFilter(std::string const& filter) const;

    int getTotalLikes() const;
};
