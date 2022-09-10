#pragma once

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include "Definitions.h"

enum class ParserTask
{
    Encode,
    Decode
};

class JsonParser
{
public:
    template <typename T>
    static void encodeDecode(
        boost::property_tree::ptree& tree,
        T& parameter,
        T const& defaultValue,
        std::string const& node,
        ParserTask task);
};

/**
 * Implementations
 */

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

template <typename T>
void JsonParser::encodeDecode(
    boost::property_tree::ptree& tree,
    T& parameter,
    T const& defaultValue,
    std::string const& node,
    ParserTask task)
{
    if (ParserTask::Encode == task) {
        if constexpr (std::is_same<T, bool>::value) {
            tree.put(node, parameter ? "true" : "false");
        } else if constexpr (std::is_same<T, std::string>::value) {
            tree.put(node, parameter);
        } else {
            tree.put(node, to_string_with_precision(parameter, 8));
        }
    } else {
        if constexpr (std::is_same<T, std::string>::value) {
            parameter = tree.get<std::string>(node, defaultValue);
            boost::algorithm::to_lower(parameter);
        } else {
            parameter = tree.get<T>(node, defaultValue);
        }
    }
}
