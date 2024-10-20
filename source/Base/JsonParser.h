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

private:
    template <typename T>
    static std::string toStringWithPrecision(T const& value, int n = 6);
};

/**
 * Implementations
 */

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
                return toStringWithPrecision(value, 8);
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
        size_t lastDotIndex = node.find_last_of('.');
        if (lastDotIndex == std::string::npos) {
            return true;
        }
        std::string path = node.substr(0, lastDotIndex);
        std::string property = node.substr(lastDotIndex + 1);
        auto subtree = tree.get_child_optional(path);
        if (!subtree) {
            return true;
        }

        return subtree->find(property) == subtree->not_found();
    }
}

template <typename T>
std::string JsonParser::toStringWithPrecision(T const& value, int n)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << value;
    return out.str();
}
