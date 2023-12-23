#pragma once

#include <map>
#include <string>
#include <variant>

#include "Definitions.h"
#include "NetworkResourceRawTO.h"

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

enum class FolderSymbols
{
    ExpandedFolder,
    ContinueFolder,
    BranchFolder,
    EndFolder,
    None
};

struct _NetworkResourceTreeTO
{
    NetworkResourceType type;
    std::vector<std::string> folderNames;
    std::vector<FolderSymbols> folderLines;
    std::variant<BrowserFolder, BrowserLeaf> node;

    bool isLeaf();
    BrowserLeaf& getLeaf();
    BrowserFolder& getFolder();
};
