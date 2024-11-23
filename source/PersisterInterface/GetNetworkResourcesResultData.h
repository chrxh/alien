#pragma once

#include "Network/NetworkResourceRawTO.h"
#include "Network/UserTO.h"

struct GetNetworkResourcesResultData
{
    std::vector<NetworkResourceRawTO> resourceTOs;
    std::vector<UserTO> userTOs;
    std::unordered_map<std::string, int> emojiTypeByResourceId;
};
