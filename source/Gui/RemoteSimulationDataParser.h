#pragma once

#include <vector>
#include <boost/property_tree/json_parser.hpp>

#include "RemoteSimulationData.h"

class RemoteSimulationDataParser
{
public:
    static std::vector<RemoteSimulationData> decode(boost::property_tree::ptree tree);
};