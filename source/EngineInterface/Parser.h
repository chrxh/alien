#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Definitions.h"
#include "DllExport.h"

class Parser
{
public:
    ENGINEINTERFACE_EXPORT static boost::property_tree::ptree encode(SimulationParameters const& parameters);
    ENGINEINTERFACE_EXPORT static SimulationParameters
    decodeSimulationParameters(boost::property_tree::ptree const& tree);

    ENGINEINTERFACE_EXPORT static boost::property_tree::ptree encode(GeneralSettings const& settings);
    ENGINEINTERFACE_EXPORT static GeneralSettings decodeGeneralSettings(boost::property_tree::ptree const& tree);
};
