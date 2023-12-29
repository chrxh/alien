#pragma once

#include <map>
#include <string>
#include <variant>

#include "Definitions.h"
#include "NetworkResourceRawTO.h"

struct BrowserFolder
{
    int numLeafs;
    int numReactions;
};

struct BrowserLeaf
{
    std::string leafName;
    NetworkResourceRawTO rawTO;
};

enum class FolderTreeSymbols
{
    Collapsed,
    Expanded,
    Continue,
    Branch,
    End,
    None
};

struct _NetworkResourceTreeTO
{
    NetworkResourceType type;
    std::vector<std::string> folderNames;
    std::vector<FolderTreeSymbols> treeSymbols;
    std::variant<BrowserFolder, BrowserLeaf> node;

    bool isLeaf();
    BrowserLeaf& getLeaf();
    BrowserFolder& getFolder();
};
