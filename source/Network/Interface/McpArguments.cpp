#include "McpArguments.h"

#include <cstdint>
#include <stdexcept>
#include <charconv>
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

namespace
{
    int parseInt(boost::json::value const& value, std::string_view key, int min, int max)
    {
        auto result = [&] {
            try {
                return value.to_number<int64_t>();
            } catch (...) {
                throw std::invalid_argument(std::format("'{}' must be an integer.", key));
            }
        }();
        return static_cast<int>(checkRange<int64_t>(result, key, min, max));
    }
}

int McpArguments::getInt(boost::json::object const& arguments, std::string_view key, int min, int max)
{
    return parseInt(getRequiredValue(arguments, key), key, min, max);
}

std::optional<std::vector<int>> McpArguments::getOptionalInts(boost::json::object const& arguments, std::string_view key, size_t minNumInts, int min, int max)
{
    if (!arguments.contains(key)) {
        return std::nullopt;
    }
    return getInts(arguments, key, minNumInts, min, max);
}

std::vector<int> McpArguments::getInts(boost::json::object const& arguments, std::string_view key, size_t minNumInts, int min, int max)
{
    auto const& value = getRequiredValue(arguments, key);
    if (!value.is_array()) {
        throw std::invalid_argument(std::format("'{}' must be an array of integers.", key));
    }
    std::vector<int> result;
    for (auto const& element : value.as_array()) {
        result.emplace_back(parseInt(element, key, min, max));
    }
    if (result.size() < minNumInts) {
        throw std::invalid_argument(std::format("'{}' needs at least {} integers.", key, minNumInts));
    }
    return result;
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

namespace
{
    uint64_t toId(boost::json::value const& value, std::string_view key)
    {
        auto invalidId = std::invalid_argument(std::format("'{}' must be an id given as a string of decimal digits.", key));
        if (value.is_string()) {
            auto const& text = value.as_string();
            uint64_t result = 0;
            auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
            if (text.empty() || error != std::errc() || end != text.data() + text.size()) {
                throw invalidId;
            }
            return result;
        }
        if (value.is_uint64()) {
            return value.as_uint64();
        }
        if (value.is_int64() && value.as_int64() >= 0) {
            return static_cast<uint64_t>(value.as_int64());
        }
        throw invalidId;
    }
}

uint64_t McpArguments::getId(boost::json::object const& arguments, std::string_view key)
{
    return toId(getRequiredValue(arguments, key), key);
}

std::optional<uint64_t> McpArguments::getOptionalId(boost::json::object const& arguments, std::string_view key)
{
    if (!arguments.contains(key)) {
        return std::nullopt;
    }
    return getId(arguments, key);
}

std::vector<uint64_t> McpArguments::getIds(boost::json::object const& arguments, std::string_view key, size_t minNumIds)
{
    auto const& value = getRequiredValue(arguments, key);
    if (!value.is_array()) {
        throw std::invalid_argument(std::format("'{}' must be an array of ids.", key));
    }
    std::vector<uint64_t> result;
    for (auto const& element : value.as_array()) {
        result.emplace_back(toId(element, key));
    }
    if (result.size() < minNumIds) {
        throw std::invalid_argument(std::format("'{}' needs at least {} ids.", key, minNumIds));
    }
    return result;
}

boost::json::value const& McpArguments::getValue(boost::json::object const& arguments, std::string_view key)
{
    return getRequiredValue(arguments, key);
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
