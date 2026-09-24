#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <boost/json/object.hpp>

// Builds JSON schemas for the input of MCP tools
class McpSchema
{
public:
    static boost::json::object object(boost::json::object properties = {}, std::vector<std::string> const& required = {});

    static boost::json::object number(std::string const& description, std::optional<double> min = std::nullopt, std::optional<double> max = std::nullopt);
    static boost::json::object integer(std::string const& description, std::optional<int64_t> min = std::nullopt, std::optional<int64_t> max = std::nullopt);
    static boost::json::object boolean(std::string const& description);
    static boost::json::object string(std::string const& description);
    static boost::json::object enumeration(std::string const& description, std::vector<std::string> const& values);
    static boost::json::object points(std::string const& description, size_t minItems);
    static boost::json::object array(std::string const& description, boost::json::object items, size_t minItems = 0);
    static boost::json::object any(std::string const& description);
};
