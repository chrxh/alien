#pragma once

#include <boost/property_tree/ptree.hpp>

#include "Definitions.h"
#include "DllExport.h"

class Parser
{
public:
    ENGINEINTERFACE_EXPORT static boost::property_tree::ptree encode(uint64_t timestep, Settings parameters);
    ENGINEINTERFACE_EXPORT static std::pair<uint64_t, Settings> decodeTimestepAndSettings(
        boost::property_tree::ptree tree);

    enum class Task
    {
        Encode,
        Decode
    };
    template <typename T>
    static void encodeDecode(
        boost::property_tree::ptree& tree,
        T& parameter,
        T const& defaultValue,
        std::string const& node,
        Task task);

private:
    static void encodeDecode(boost::property_tree::ptree& tree, uint64_t& timestep, Settings& settings, Task task);
};

/**
 * Implementations
 */
template <typename T>
void Parser::encodeDecode(
    boost::property_tree::ptree& tree,
    T& parameter,
    T const& defaultValue,
    std::string const& node,
    Task task)
{
    if (Task::Encode == task) {
        if constexpr (std::is_same<T, bool>::value) {
            tree.add(node, parameter ? "true" : "false");
        } else {
            tree.add(node, std::to_string(parameter));
        }
    } else {
        parameter = tree.get<T>(node, defaultValue);
    }
}
