#pragma once

#include <string>

class ImGuiTableSortSpecs;

enum RemoteSimulationDataColumnId
{
    RemoteSimulationDataColumnId_Timestamp,
    RemoteSimulationDataColumnId_UserName,
    RemoteSimulationDataColumnId_SimulationName,
    RemoteSimulationDataColumnId_Likes,
    RemoteSimulationDataColumnId_Width,
    RemoteSimulationDataColumnId_Height,
    RemoteSimulationDataColumnId_Size,
    RemoteSimulationDataColumnId_Description,
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
    int width;
    int height;
    uint64_t contentSize;
    std::string description;
    std::string version;

    static int compare(void const* left, void const* right, ImGuiTableSortSpecs const* specs);
};
