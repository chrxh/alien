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
    //returns true if defaultValue has been applied
    template <typename T>
    static bool encodeDecode(boost::property_tree::ptree& tree, T& value, T const& defaultValue, std::string const& node, ParserTask task);
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
bool JsonParser::encodeDecode(
    boost::property_tree::ptree& tree,
    T& value,
    T const& defaultValue,
    std::string const& node,
    ParserTask task)
{
    if (ParserTask::Encode == task) {
        std::string stringValue = [&] {
            if constexpr (std::is_same<T, bool>::value) {
                return value ? std::string("true") : std::string("false");
            } else if constexpr (std::is_same<T, std::string>::value) {
                return value;
            } else {
                return to_string_with_precision(value, 8);
            }
        }();
        tree.put(node, stringValue);
        return false;
    } else {
        if constexpr (std::is_same<T, std::string>::value) {
            value = tree.get<std::string>(node, defaultValue);
        } else {
            value = tree.get<T>(node, defaultValue);
        }
        return tree.find(node) == tree.not_found();
    }
}
