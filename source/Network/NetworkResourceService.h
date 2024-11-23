#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "Definitions.h"

class NetworkResourceService
{
    MAKE_SINGLETON(NetworkResourceService);

public:
    std::vector<NetworkResourceTreeTO> createTreeTOs(
        std::vector<NetworkResourceRawTO> const& rawTOs,
        std::set<std::vector<std::string>> const& collapsedFolderNames);

    std::vector<NetworkResourceRawTO> getMatchingRawTOs(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs);
    void invalidateCache();  //invalidate cache for getMatchingRawTOs

    //folder names conversion methods
    std::vector<std::string> getFolderNames(std::string const& resourceName);
    std::string removeFoldersFromName(std::string const& resourceName);
    std::set<std::vector<std::string>> getFolderNames(std::vector<NetworkResourceRawTO> const& browserData, int minNesting = 2);
    std::string concatenateFolderName(std::vector<std::string> const& folderNames, bool withSlashAtTheEnd);
    std::vector<std::string> convertFolderNamesToSettings(std::set<std::vector<std::string>> const& folderNames);
    std::set<std::vector<std::string>> convertSettingsToFolderNames(std::vector<std::string> const& settings);

private:
    std::unordered_map<NetworkResourceTreeTO, std::vector<NetworkResourceRawTO>> _treeTOtoRawTOcache;
};
