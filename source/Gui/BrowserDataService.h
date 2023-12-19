#pragma once

#include <vector>

#include "Definitions.h"
#include "RemoteSimulationData.h"

class BrowserDataService
{
public:
    static std::vector<BrowserSimulationData> createBrowserData(std::vector<RemoteSimulationData> const& remoteData);
};
