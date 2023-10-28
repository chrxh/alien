#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Base/JsonParser.h"
#include "EngineInterface/SimulationParametersSpotValues.h"

#include "AuxiliaryData.h"
#include "Definitions.h"

class AuxiliaryDataParserService
{
public:
    static boost::property_tree::ptree encodeAuxiliaryData(AuxiliaryData const& data);
    static AuxiliaryData decodeAuxiliaryData(boost::property_tree::ptree tree);

    static boost::property_tree::ptree encodeSimulationParameters(SimulationParameters const& data);
    static SimulationParameters decodeSimulationParameters(boost::property_tree::ptree tree);
};
