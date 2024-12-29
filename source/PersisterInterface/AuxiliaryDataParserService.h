#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Base/JsonParser.h"
#include "Base/Singleton.h"
#include "EngineInterface/SimulationParametersZoneValues.h"

#include "AuxiliaryData.h"
#include "Definitions.h"

class AuxiliaryDataParserService
{
    MAKE_SINGLETON(AuxiliaryDataParserService);

public:
    boost::property_tree::ptree encodeAuxiliaryData(AuxiliaryData const& data);
    AuxiliaryData decodeAuxiliaryData(boost::property_tree::ptree tree);

    boost::property_tree::ptree encodeSimulationParameters(SimulationParameters const& data);
    SimulationParameters decodeSimulationParameters(boost::property_tree::ptree tree);
};
