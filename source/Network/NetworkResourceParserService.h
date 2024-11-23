#pragma once

#include <vector>
#include <boost/property_tree/json_parser.hpp>

#include "Base/Singleton.h"

#include "UserTO.h"
#include "Definitions.h"

class NetworkResourceParserService
{
    MAKE_SINGLETON(NetworkResourceParserService);

public:
    std::vector<NetworkResourceRawTO> decodeRemoteSimulationData(boost::property_tree::ptree const& tree);
    std::vector<UserTO> decodeUserData(boost::property_tree::ptree const& tree);
};