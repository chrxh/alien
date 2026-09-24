#include "McpSchema.h"

#include <boost/json.hpp>

boost::json::object McpSchema::object(boost::json::object properties, std::vector<std::string> const& required)
{
    boost::json::object result{{"type", "object"}, {"properties", std::move(properties)}};
    if (!required.empty()) {
        boost::json::array requiredArray;
        for (auto const& name : required) {
            requiredArray.emplace_back(name);
        }
        result["required"] = std::move(requiredArray);
    }
    return result;
}

boost::json::object McpSchema::number(std::string const& description, std::optional<double> min, std::optional<double> max)
{
    boost::json::object result{{"type", "number"}, {"description", description}};
    if (min) {
        result["minimum"] = *min;
    }
    if (max) {
        result["maximum"] = *max;
    }
    return result;
}

boost::json::object McpSchema::integer(std::string const& description, std::optional<int64_t> min, std::optional<int64_t> max)
{
    boost::json::object result{{"type", "integer"}, {"description", description}};
    if (min) {
        result["minimum"] = *min;
    }
    if (max) {
        result["maximum"] = *max;
    }
    return result;
}

boost::json::object McpSchema::boolean(std::string const& description)
{
    return {{"type", "boolean"}, {"description", description}};
}

boost::json::object McpSchema::string(std::string const& description)
{
    return {{"type", "string"}, {"description", description}};
}

boost::json::object McpSchema::enumeration(std::string const& description, std::vector<std::string> const& values)
{
    boost::json::array valueArray;
    for (auto const& value : values) {
        valueArray.emplace_back(value);
    }
    return {{"type", "string"}, {"description", description}, {"enum", std::move(valueArray)}};
}

boost::json::object McpSchema::points(std::string const& description, size_t minItems)
{
    return {
        {"type", "array"},
        {"description", description},
        {"minItems", minItems},
        {"items", boost::json::object{{"type", "array"}, {"items", boost::json::object{{"type", "number"}}}, {"minItems", 2}, {"maxItems", 2}}},
    };
}
