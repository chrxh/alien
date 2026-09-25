#include "McpArguments.h"

#include <cstdint>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

namespace
{
    boost::json::value const& getRequiredValue(boost::json::object const& arguments, std::string_view key)
    {
        auto value = arguments.if_contains(key);
        if (!value) {
            throw std::invalid_argument(std::format("Missing argument '{}'.", key));
        }
        return *value;
    }

    template <typename T>
    T checkRange(T value, std::string_view key, T min, T max)
    {
        if (value < min || value > max) {
            throw std::invalid_argument(std::format("'{}' must be between {} and {}.", key, min, max));
        }
        return value;
    }
}

std::optional<float> McpArguments::getOptionalFloat(boost::json::object const& arguments, std::string_view key, float min, float max)
{
    if (!arguments.contains(key)) {
        return std::nullopt;
    }
    return getFloat(arguments, key, min, max);
}

float McpArguments::getFloat(boost::json::object const& arguments, std::string_view key, float min, float max)
{
    auto const& value = getRequiredValue(arguments, key);
    if (!value.is_number()) {
        throw std::invalid_argument(std::format("'{}' must be a number.", key));
    }
    return checkRange(static_cast<float>(value.to_number<double>()), key, min, max);
}

std::optional<int> McpArguments::getOptionalInt(boost::json::object const& arguments, std::string_view key, int min, int max)
{
    if (!arguments.contains(key)) {
        return std::nullopt;
    }
    return getInt(arguments, key, min, max);
}

int McpArguments::getInt(boost::json::object const& arguments, std::string_view key, int min, int max)
{
    auto const& value = getRequiredValue(arguments, key);
    auto result = [&] {
        try {
            return value.to_number<int64_t>();
        } catch (...) {
            throw std::invalid_argument(std::format("'{}' must be an integer.", key));
        }
    }();
    return static_cast<int>(checkRange<int64_t>(result, key, min, max));
}

std::optional<bool> McpArguments::getOptionalBool(boost::json::object const& arguments, std::string_view key)
{
    if (!arguments.contains(key)) {
        return std::nullopt;
    }
    return getBool(arguments, key);
}

bool McpArguments::getBool(boost::json::object const& arguments, std::string_view key)
{
    auto const& value = getRequiredValue(arguments, key);
    if (!value.is_bool()) {
        throw std::invalid_argument(std::format("'{}' must be a boolean.", key));
    }
    return value.as_bool();
}

std::optional<std::string> McpArguments::getOptionalString(boost::json::object const& arguments, std::string_view key)
{
    if (!arguments.contains(key)) {
        return std::nullopt;
    }
    return getString(arguments, key);
}

std::string McpArguments::getString(boost::json::object const& arguments, std::string_view key)
{
    auto const& value = getRequiredValue(arguments, key);
    if (!value.is_string()) {
        throw std::invalid_argument(std::format("'{}' must be a string.", key));
    }
    return std::string(value.as_string());
}

std::filesystem::path McpArguments::getFilePath(boost::json::object const& arguments, std::string_view key)
{
    auto value = getString(arguments, key);
    return std::filesystem::path(std::u8string(value.begin(), value.end()));
}

std::vector<RealVector2D> McpArguments::getPoints(boost::json::object const& arguments, std::string_view key, size_t minNumPoints)
{
    auto const& value = getRequiredValue(arguments, key);
    auto invalidPoints = std::invalid_argument(std::format("'{}' must be an array of [x, y] pairs.", key));
    if (!value.is_array()) {
        throw invalidPoints;
    }
    std::vector<RealVector2D> result;
    for (auto const& point : value.as_array()) {
        if (!point.is_array() || point.as_array().size() != 2 || !point.as_array().at(0).is_number() || !point.as_array().at(1).is_number()) {
            throw invalidPoints;
        }
        result.emplace_back(
            RealVector2D{static_cast<float>(point.as_array().at(0).to_number<double>()), static_cast<float>(point.as_array().at(1).to_number<double>())});
    }
    if (result.size() < minNumPoints) {
        throw std::invalid_argument(std::format("'{}' needs at least {} points.", key, minNumPoints));
    }
    return result;
}

std::vector<boost::json::object> McpArguments::getObjects(boost::json::object const& arguments, std::string_view key, size_t minNumObjects)
{
    auto const& value = getRequiredValue(arguments, key);
    auto invalidObjects = std::invalid_argument(std::format("'{}' must be an array of objects.", key));
    if (!value.is_array()) {
        throw invalidObjects;
    }
    std::vector<boost::json::object> result;
    for (auto const& element : value.as_array()) {
        if (!element.is_object()) {
            throw invalidObjects;
        }
        result.emplace_back(element.as_object());
    }
    if (result.size() < minNumObjects) {
        throw std::invalid_argument(std::format("'{}' needs at least {} entries.", key, minNumObjects));
    }
    return result;
}
