#pragma once

#include <map>
#include <string>
#include <variant>

#include "Definitions.h"

using BrowserDataType = int;
enum BrowserDataType_
{
    BrowserDataType_Simulation,
    BrowserDataType_Genome
};

struct BrowserFolder
{
};

struct BrowserLeaf
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
};

struct _BrowserDataTO
{
    BrowserDataType type;
    std::vector<std::string> folders;
    std::variant<BrowserFolder, BrowserLeaf> node;

    bool isLeaf();
    BrowserLeaf& getLeaf();
    BrowserFolder& getFolder();
};
