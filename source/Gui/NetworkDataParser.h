#pragma once

#include <vector>
#include <boost/property_tree/json_parser.hpp>

#include "RemoteSimulationData.h"
#include "UserData.h"

class NetworkDataParser
{
public:
    static std::vector<RemoteSimulationData> decodeRemoteSimulationData(boost::property_tree::ptree tree);
    static std::vector<UserData> decodeUserData(boost::property_tree::ptree tree);
};