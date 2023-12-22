#pragma once

#include <vector>
#include <boost/property_tree/json_parser.hpp>

#include "UserTO.h"
#include "Definitions.h"

class NetworkResourceParserService
{
public:
    static std::vector<NetworkResourceRawTO> decodeRemoteSimulationData(boost::property_tree::ptree const& tree);
    static std::vector<UserTO> decodeUserData(boost::property_tree::ptree const& tree);
};