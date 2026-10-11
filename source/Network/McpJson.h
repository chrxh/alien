#pragma once

#include <string>

#include <boost/json/value.hpp>

class McpJson
{
public:
    static std::string serialize(boost::json::value const& value);
};
