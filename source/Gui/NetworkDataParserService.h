#pragma once

#include <vector>
#include <boost/property_tree/json_parser.hpp>

#include "UserData.h"
#include "Definitions.h"

class NetworkDataParserService
{
public:
    static std::vector<NetworkDataTO> decodeRemoteSimulationData(boost::property_tree::ptree const& tree);
    static std::vector<UserData> decodeUserData(boost::property_tree::ptree const& tree);
};