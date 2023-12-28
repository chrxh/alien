#pragma once

#include <vector>

#include "Definitions.h"
#include "NetworkResourceRawTO.h"

class NetworkResourceService
{
public:
    static std::vector<NetworkResourceTreeTO> createTreeTOs(
        std::vector<NetworkResourceRawTO> const& browserData,
        std::set<std::vector<std::string>> const& collapsedFolderNames);
};
