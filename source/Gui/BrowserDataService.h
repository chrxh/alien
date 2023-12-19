#pragma once

#include <vector>

#include "Definitions.h"
#include "NetworkDataTO.h"

class BrowserDataService
{
public:
    static std::vector<BrowserDataTO> createBrowserData(std::vector<NetworkDataTO> const& remoteData);
};
