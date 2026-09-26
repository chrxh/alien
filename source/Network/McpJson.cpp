#include "McpJson.h"

#include <cmath>
#include <format>

#include <boost/json.hpp>

namespace
{
    auto constexpr MaxExactInteger = 1e15;

    void serializeDouble(std::string& output, double value)
    {
        if (!std::isfinite(value)) {
            output += "null";
        } else if (std::trunc(value) == value && std::abs(value) < MaxExactInteger) {
            output += std::format("{}", static_cast<int64_t>(value));
        } else if (static_cast<double>(static_cast<float>(value)) == value) {
            output += std::format("{}", static_cast<float>(value));
        } else {
            output += std::format("{}", value);
        }
    }

    void serializeValue(std::string& output, boost::json::value const& value)
    {
        switch (value.kind()) {
        case boost::json::kind::double_:
            serializeDouble(output, value.get_double());
            break;
        case boost::json::kind::array: {
            output += '[';
            auto first = true;
            for (auto const& element : value.get_array()) {
                if (!first) {
                    output += ',';
                }
                first = false;
                serializeValue(output, element);
            }
            output += ']';
            break;
        }
        case boost::json::kind::object: {
            output += '{';
            auto first = true;
            for (auto const& [key, element] : value.get_object()) {
                if (!first) {
                    output += ',';
                }
                first = false;
                output += boost::json::serialize(boost::json::string(key));
                output += ':';
                serializeValue(output, element);
            }
            output += '}';
            break;
        }
        default:
            output += boost::json::serialize(value);
            break;
        }
    }
}

std::string McpJson::serialize(boost::json::value const& value)
{
    std::string result;
    serializeValue(result, value);
    return result;
}
