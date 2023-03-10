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
        FloatColorVector& parameter,
        FloatColorVector const& defaultValue,
        std::string const& node,
        ParserTask task);
    template <>
    static void encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        IntColorVector& parameter,
        IntColorVector const& defaultValue,
        std::string const& node,
        ParserTask task);
    template <>
    static void encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        FloatColorMatrix& parameter,
        FloatColorMatrix const& defaultValue,
        std::string const& node,
        ParserTask task);
    template <>
    static void encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        IntColorMatrix& parameter,
        IntColorMatrix const& defaultValue,
        std::string const& node,
        ParserTask task);
    template <>
    static void encodeDecodeProperty(
        boost::property_tree::ptree& tree,
        BoolColorMatrix& parameter,
        BoolColorMatrix const& defaultValue,
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
        FloatColorVector& parameter,
        bool& isActivated,
        FloatColorVector const& defaultValue,
        std::string const& node,
        ParserTask task);
    template <>
    static void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        IntColorVector& parameter,
        bool& isActivated,
        IntColorVector const& defaultValue,
        std::string const& node,
        ParserTask task);
};