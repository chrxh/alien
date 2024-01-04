#pragma once

#include <vector>

#include "Definitions.h"

class NetworkResourceService
{
public:
    static std::vector<NetworkResourceTreeTO> createTreeTOs(
        std::vector<NetworkResourceRawTO> const& rawTOs,
        std::set<std::vector<std::string>> const& collapsedFolderNames);

    static std::vector<NetworkResourceRawTO> getMatchingRawTOs(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs);
    static void invalidateCache();  //invalidate cache for getMatchingRawTOs

    //folder names conversion methods
    static std::vector<std::string> getFolderNames(std::string const& resourceName);
    static std::string removeFoldersFromName(std::string const& resourceName);
    static std::set<std::vector<std::string>> getFolderNames(std::vector<NetworkResourceRawTO> const& browserData, int minNesting = 2);
    static std::string concatenateFolderName(std::vector<std::string> const& folderNames, bool withSlashAtTheEnd);
    static std::string convertFolderNamesToSettings(std::set<std::vector<std::string>> const& data);
    static std::set<std::vector<std::string>> convertSettingsToFolderNames(std::string const& data);

private:
    static std::unordered_map<NetworkResourceTreeTO, std::vector<NetworkResourceRawTO>> _treeTOtoRawTOcache;
};
