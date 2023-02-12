#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Base/JsonParser.h"

#include "AuxiliaryData.h"
#include "Definitions.h"

class AuxiliaryDataParser
{
public:
    static boost::property_tree::ptree encode(AuxiliaryData const& data);
    static AuxiliaryData decode(boost::property_tree::ptree tree);

private:
    static void encodeDecode(boost::property_tree::ptree& tree, AuxiliaryData& data, ParserTask task);

    template <typename T>
    static void encodeDecodeSpotProperty(
        boost::property_tree::ptree& tree,
        T& parameter,
        bool& isActivated,
        T const& defaultValue,
        std::string const& node,
        ParserTask task);
};