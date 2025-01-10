#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Base/JsonParser.h"
#include "Base/Singleton.h"
#include "EngineInterface/SimulationParametersZoneValues.h"

#include "SettingsData.h"
#include "Definitions.h"

class SettingsDataParserService
{
    MAKE_SINGLETON(SettingsDataParserService);

public:
    boost::property_tree::ptree encodeAuxiliaryData(SettingsData const& data);
    SettingsData decodeAuxiliaryData(boost::property_tree::ptree tree);

    boost::property_tree::ptree encodeSimulationParameters(SimulationParameters const& data);
    SimulationParameters decodeSimulationParameters(boost::property_tree::ptree tree);
};
