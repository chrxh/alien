#pragma once

#include <string>

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

class RemoteSimulationData
{
public:
    std::string id;
    std::string timestamp;
    std::string userName;
    std::string simName;
    int likes;
    int numDownloads;
    int width;
    int height;
    int particles;
    uint64_t contentSize;
    std::string description;
    std::string version;

    static int compare(void const* left, void const* right, ImGuiTableSortSpecs const* specs);
    bool matchWithFilter(std::string const& filter) const;
};
