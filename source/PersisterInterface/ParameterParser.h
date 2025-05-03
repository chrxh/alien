#pragma once

#include <chrono>

#include "Base/JsonParser.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationParameters.h"

class ParameterParser
{
public:

    //return true if value does not exist in tree
    template <typename T>
    static bool encodeDecode(boost::property_tree::ptree& tree, T& value, T const& defaultValue, std::string const& node, ParserTask task);
};

namespace detail
{
    template <typename T>
    static bool encodeDecodeImpl(boost::property_tree::ptree& tree, T& value, T const& defaultValue, std::string const& node, ParserTask task)
    {
        return JsonParser::encodeDecode(tree, value, defaultValue, node, task);
    }

    template <>
    inline bool encodeDecodeImpl(boost::property_tree::ptree& tree, Char64& value, Char64 const& defaultValue, std::string const& node, ParserTask task)
    {
        bool result;
        if (task == ParserTask::Encode) {
            std::string valueAsString(value);
            std::string defaultValueAsString(defaultValue);
            result = JsonParser::encodeDecode(tree, valueAsString, defaultValueAsString, node, task);
        } else {
            std::string valueAsString;
            std::string defaultValueAsString(defaultValue);
            result = JsonParser::encodeDecode(tree, valueAsString, defaultValueAsString, node, task);
            auto copyLength = std::min(63, toInt(valueAsString.size()));
            valueAsString.copy(value, copyLength);
            value[copyLength] = '\0';
        }
        return result;
    }

    template <>
    inline bool
    encodeDecodeImpl(boost::property_tree::ptree& tree, IntVector2D& value, IntVector2D const& defaultValue, std::string const& node, ParserTask task)
    {
        auto result = false;
        result |= encodeDecodeImpl(tree, value.x, defaultValue.x, node + ".X", task);
        result |= encodeDecodeImpl(tree, value.y, defaultValue.y, node + ".Y", task);
        return result;
    }

    template <>
    inline bool
    encodeDecodeImpl(boost::property_tree::ptree& tree, RealVector2D& value, RealVector2D const& defaultValue, std::string const& node, ParserTask task)
    {
        auto result = false;
        result |= encodeDecodeImpl(tree, value.x, defaultValue.x, node + ".X", task);
        result |= encodeDecodeImpl(tree, value.y, defaultValue.y, node + ".Y", task);
        return result;
    }

    template <>
    inline bool
    encodeDecodeImpl(boost::property_tree::ptree& tree, FloatColorRGB& value, FloatColorRGB const& defaultValue, std::string const& node, ParserTask task)
    {
        auto result = false;
        result |= encodeDecodeImpl(tree, value.r, defaultValue.r, node + ".R", task);
        result |= encodeDecodeImpl(tree, value.g, defaultValue.g, node + ".G", task);
        result |= encodeDecodeImpl(tree, value.b, defaultValue.b, node + ".B", task);
        return result;
    }

    template <>
    inline bool encodeDecodeImpl(
        boost::property_tree::ptree& tree,
        ColorTransitionRule& value,
        ColorTransitionRule const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        result |= encodeDecodeImpl(tree, value.duration, defaultValue.duration, node + ".Duration", task);
        result |= encodeDecodeImpl(tree, value.targetColor, defaultValue.targetColor, node + ".Target", task);
        return result;
    }

    template <>
    inline  bool encodeDecodeImpl(
        boost::property_tree::ptree& tree,
        std::chrono::milliseconds& parameter,
        std::chrono::milliseconds const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        if (ParserTask::Encode == task) {
            auto parameterAsString = std::to_string(parameter.count());
            return encodeDecodeImpl(tree, parameterAsString, std::string(), node, task);
        } else {
            std::string parameterAsString;
            auto defaultAsString = std::to_string(defaultValue.count());
            auto result = encodeDecodeImpl(tree, parameterAsString, defaultAsString, node, task);
            parameter = std::chrono::milliseconds(std::stoi(parameterAsString));
            return result;
        }
    }
};

template <typename T>
bool ParameterParser::encodeDecode(boost::property_tree::ptree& tree, T& value, T const& defaultValue, std::string const& node, ParserTask task)
{
    return detail::encodeDecodeImpl(tree, value, defaultValue, node, task);
}
