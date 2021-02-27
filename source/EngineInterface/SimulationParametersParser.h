#pragma once

#include "Definitions.h"

class ENGINEINTERFACE_EXPORT SimulationParametersParser
{
public:
    static boost::property_tree::ptree encode(SimulationParameters const& parameters);
    static SimulationParameters decode(boost::property_tree::ptree const& tree);
};