#pragma once

#include <chrono>

#include "Base/JsonParser.h"
#include "EngineInterface/Colors.h"

class PropertyParser
{
public:

    //return true if value does not exist in tree
    template <typename T>
    static bool encodeDecode(boost::property_tree::ptree& tree, T& parameter, T const& defaultValue, std::string const& node, ParserTask task);

    template <typename T>
    static bool encodeDecodeWithEnabled(
        boost::property_tree::ptree& tree,
        T& parameter,
        bool& isActivated,
        T const& defaultValue,
        std::string const& node,
        ParserTask task);
};

namespace detail
{
    template <typename T>
    static bool encodeDecodeImpl(boost::property_tree::ptree& tree, T& parameter, T const& defaultValue, std::string const& node, ParserTask task)
    {
        return JsonParser::encodeDecode(tree, parameter, defaultValue, node, task);
    }

    template <>
    inline bool encodeDecodeImpl(
        boost::property_tree::ptree& tree,
        ColorVector<float>& parameter,
        ColorVector<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result |= encodeDecodeImpl(tree, parameter[i], defaultValue[i], node + "[" + std::to_string(i) + "]", task);
        }
        return result;
    }

    template <>
    inline  bool encodeDecodeImpl(
        boost::property_tree::ptree& tree,
        ColorVector<int>& parameter,
        ColorVector<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result |= encodeDecodeImpl(tree, parameter[i], defaultValue[i], node + "[" + std::to_string(i) + "]", task);
        }
        return result;
    }

    template <>
    inline  bool encodeDecodeImpl<ColorMatrix<float>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<float>& parameter,
        ColorMatrix<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                result |= encodeDecodeImpl(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
        return result;
    }

    template <>
    inline  bool encodeDecodeImpl<ColorMatrix<int>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<int>& parameter,
        ColorMatrix<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                result |= encodeDecodeImpl(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
        return result;
    }

    template <>
    inline  bool encodeDecodeImpl<ColorMatrix<bool>>(
        boost::property_tree::ptree& tree,
        ColorMatrix<bool>& parameter,
        ColorMatrix<bool> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = false;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++j) {
                result |= encodeDecodeImpl(tree, parameter[i][j], defaultValue[i][j], node + "[" + std::to_string(i) + ", " + std::to_string(j) + "]", task);
            }
        }
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

    template <typename T>
    inline  bool encodeDecodeWithEnabledImpl(
        boost::property_tree::ptree& tree,
        T& parameter,
        bool& isActivated,
        T const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = encodeDecodeImpl(tree, isActivated, false, node + ".activated", task);
        result |= encodeDecodeImpl(tree, parameter, defaultValue, node + ".value", task);
        return result;
    }

    template <>
    inline  bool encodeDecodeWithEnabledImpl(
        boost::property_tree::ptree& tree,
        ColorVector<float>& parameter,
        bool& isActivated,
        ColorVector<float> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = encodeDecodeImpl(tree, isActivated, false, node + ".activated", task);
        result |= encodeDecodeImpl(tree, parameter, defaultValue, node, task);
        return result;
    }

    template <>
    inline  bool encodeDecodeWithEnabledImpl(
        boost::property_tree::ptree& tree,
        ColorVector<int>& parameter,
        bool& isActivated,
        ColorVector<int> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = encodeDecodeImpl(tree, isActivated, false, node + ".activated", task);
        result |= encodeDecodeImpl(tree, parameter, defaultValue, node, task);
        return result;
    }

    template <typename T>
    inline  bool encodeDecodeWithEnabledImpl(
        boost::property_tree::ptree& tree,
        ColorMatrix<T>& parameter,
        bool& isActivated,
        ColorMatrix<bool> const& defaultValue,
        std::string const& node,
        ParserTask task)
    {
        auto result = encodeDecodeImpl(tree, isActivated, false, node + ".activated", task);
        result |= encodeDecodeImpl(tree, parameter, defaultValue, node, task);
        return result;
    }
};

template <typename T>
bool PropertyParser::encodeDecode(boost::property_tree::ptree& tree, T& parameter, T const& defaultValue, std::string const& node, ParserTask task)
{
    return detail::encodeDecodeImpl(tree, parameter, defaultValue, node, task);
}

template <typename T>
bool PropertyParser::encodeDecodeWithEnabled(
    boost::property_tree::ptree& tree,
    T& parameter,
    bool& isActivated,
    T const& defaultValue,
    std::string const& node,
    ParserTask task)
{
    return detail::encodeDecodeWithEnabledImpl(tree, parameter, isActivated, defaultValue, node, task);
}
