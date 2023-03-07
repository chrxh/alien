#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Base/JsonParser.h"
#include "EngineInterface/SimulationParametersSpotValues.h"

#include "AuxiliaryData.h"
#include "Definitions.h"

class AuxiliaryDataParser
{
public:
    static boost::property_tree::ptree encodeAuxiliaryData(AuxiliaryData const& data);
    static AuxiliaryData decodeAuxiliaryData(boost::property_tree::ptree tree);

    static boost::property_tree::ptree encodeSimulationParameters(SimulationParameters const& data);
    static SimulationParameters decodeSimulationParameters(boost::property_tree::ptree tree);

private:
    static void encodeDecode(boost::property_tree::ptree& tree, AuxiliaryData& data, ParserTask task);
    static void encodeDecode(boost::property_tree::ptree& tree, SimulationParameters& parameters, ParserTask task);

    template <typename T>
    static void encodeDecodeProperty(boost::property_tree::ptree& tree, T& parameter, T const& defaultValue, std::string const& node, ParserTask task);
    template <>
    static void encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        FloatByColor& parameter,
        FloatByColor const& defaultValue,
        std::string const& node,
        ParserTask task);
    template <>
    static void encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        IntByColor& parameter,
        IntByColor const& defaultValue,
        std::string const& node,
        ParserTask task);

    template <typename T>
    static void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        T& parameter,
        bool& isActivated,
        T const& defaultValue,
        std::string const& node,
        ParserTask task);
    template <>
    static void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        FloatByColor& parameter,
        bool& isActivated,
        FloatByColor const& defaultValue,
        std::string const& node,
        ParserTask task);
    template <>
    static void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        IntByColor& parameter,
        bool& isActivated,
        IntByColor const& defaultValue,
        std::string const& node,
        ParserTask task);
};