#pragma once

#include <boost/property_tree/ptree.hpp>

#include <Base/JsonParser.h>
#include <Base/Singleton.h>

#include <EngineInterface/SimulationParameters.h>

#include "Definitions.h"

class SettingsParserService
{
    MAKE_SINGLETON(SettingsParserService);

public:
    boost::property_tree::ptree encodeSimulationParameters(SimulationParameters const& data);
    SimulationParameters decodeSimulationParameters(boost::property_tree::ptree tree);
};
