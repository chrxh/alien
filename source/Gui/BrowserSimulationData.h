#pragma once

#include <map>
#include <string>

using BrowserDataType = int;
enum BrowserDataType_
{
    BrowserDataType_Simulation,
    BrowserDataType_Genome
};

class _BrowserSimulationData
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
    BrowserDataType type;
};
        
