#pragma once

#include <Network/Interface/NetworkResourceRawTO.h>
#include <Network/Interface/UserTO.h>

struct GetNetworkResourcesResultData
{
    std::vector<NetworkResourceRawTO> resourceTOs;
    std::vector<UserTO> userTOs;
    std::unordered_map<std::string, int> emojiTypeByResourceId;
};
