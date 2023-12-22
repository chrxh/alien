#pragma once

#include <vector>

#include "Definitions.h"
#include "NetworkResourceRawTO.h"

class NetworkResourceService
{
public:
    static std::vector<NetworkResourceTreeTO> createBrowserData(std::vector<NetworkResourceRawTO> const& browserData);
};
