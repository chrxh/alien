#pragma once

#include <vector>

#include "Definitions.h"
#include "NetworkDataTO.h"

class BrowserDataService
{
public:
    static std::vector<BrowserSimulationData> createBrowserData(std::vector<NetworkDataTO> const& remoteData);
};
