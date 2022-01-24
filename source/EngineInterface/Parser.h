#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Base/JsonParser.h"

#include "Definitions.h"

class Parser
{
public:
    static boost::property_tree::ptree encode(uint64_t timestep, Settings parameters);
    static std::pair<uint64_t, Settings> decodeTimestepAndSettings(
        boost::property_tree::ptree tree);

private:
    static void
    encodeDecode(boost::property_tree::ptree& tree, uint64_t& timestep, Settings& settings, ParserTask task);
};