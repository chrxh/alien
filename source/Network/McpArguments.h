#pragma once

#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <boost/json/object.hpp>

#include <Base/Definitions.h>

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

    static std::filesystem::path getFilePath(boost::json::object const& arguments, std::string_view key);

    static std::vector<RealVector2D> getPoints(boost::json::object const& arguments, std::string_view key, size_t minNumPoints);

    static std::vector<boost::json::object> getObjects(boost::json::object const& arguments, std::string_view key, size_t minNumObjects);
};
