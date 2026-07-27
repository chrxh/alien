#pragma once

#include <boost/property_tree/ptree.hpp>

#include <Base/JsonParser.h>
#include <Base/Singleton.h>

#include "Definitions.h"
#include "SettingsForSerialization.h"

class SettingsParserService
{
    MAKE_SINGLETON(SettingsParserService);

public:
    boost::property_tree::ptree encodeSimulationParameters(SimulationParameters const& data);
    SimulationParameters decodeSimulationParameters(boost::property_tree::ptree tree);

    // Older versions stored the general settings in the settings file instead of the simulation file
    void decodeLegacyGeneralSettings(SettingsForSerialization& data, boost::property_tree::ptree tree);
};
