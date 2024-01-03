#pragma once

#include <vector>

#include "Definitions.h"

class NetworkResourceService
{
public:
    static std::vector<NetworkResourceTreeTO> createTreeTOs(
        std::vector<NetworkResourceRawTO> const& rawTOs,
        std::set<std::vector<std::string>> const& collapsedFolderNames);

    static std::vector<NetworkResourceRawTO> getAllRawTOs(
        NetworkResourceTreeTO const& treeTO,
        std::vector<NetworkResourceRawTO> const& rawTOs,
        std::unordered_map<NetworkResourceRawTO, size_t> const& indices);
    static std::set<std::vector<std::string>> getAllFolderNames(std::vector<NetworkResourceRawTO> const& browserData, int minNesting = 2);

    static std::string concatenateFolderNames(std::vector<std::string> const& folderNames, bool withSlash);
    static std::string convertFolderNamesToSettings(std::set<std::vector<std::string>> const& data);
    static std::set<std::vector<std::string>> convertSettingsToFolderNames(std::string const& data);
};
