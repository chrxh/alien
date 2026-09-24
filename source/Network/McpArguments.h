#pragma once

#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <boost/json/object.hpp>

#include <Base/Definitions.h>

// Reads tool arguments of MCP requests. Invalid arguments throw std::invalid_argument with a message for the MCP client.
class McpArguments
{
public:
    static std::optional<float> getOptionalFloat(
        boost::json::object const& arguments,
        std::string_view key,
        float min = std::numeric_limits<float>::lowest(),
        float max = std::numeric_limits<float>::max());
    static float getFloat(
        boost::json::object const& arguments,
        std::string_view key,
        float min = std::numeric_limits<float>::lowest(),
        float max = std::numeric_limits<float>::max());

    static std::optional<int> getOptionalInt(
        boost::json::object const& arguments,
        std::string_view key,
        int min = std::numeric_limits<int>::lowest(),
        int max = std::numeric_limits<int>::max());
    static int
    getInt(boost::json::object const& arguments, std::string_view key, int min = std::numeric_limits<int>::lowest(), int max = std::numeric_limits<int>::max());

    static std::optional<bool> getOptionalBool(boost::json::object const& arguments, std::string_view key);
    static bool getBool(boost::json::object const& arguments, std::string_view key);

    static std::optional<std::string> getOptionalString(boost::json::object const& arguments, std::string_view key);
    static std::string getString(boost::json::object const& arguments, std::string_view key);

    // Expects an array of [x, y] pairs
    static std::vector<RealVector2D> getPoints(boost::json::object const& arguments, std::string_view key, size_t minNumPoints);
};
