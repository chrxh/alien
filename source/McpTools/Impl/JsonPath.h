#pragma once

#include <string>
#include <vector>

#include <boost/json/value.hpp>

// Path to values inside a JSON document, e.g. "cellType.sensor.maxRange" or "neuralNetwork.biases[*]".
// '[n]' selects an array element, '[*]' all array elements and '*' all fields or alternatives of an object.
class JsonPath
{
public:
    struct Segment
    {
        enum class Type
        {
            Key,
            Index,
            AnyKey,
            AnyIndex
        };
        Type type = Type::Key;
        std::string key;
        int index = 0;
    };

    JsonPath() = default;
    explicit JsonPath(std::string const& text);

    std::string const& getText() const { return _text; }
    std::vector<Segment> const& getSegments() const { return _segments; }

    // The format is a description as returned by SerializerService::getJsonFormatOfGenome, where alternatives are listed under 'oneOf'
    // and arrays show one example element. Throws std::invalid_argument if the path does not exist in the format.
    std::vector<boost::json::value const*> resolveFormat(boost::json::value const& format) const;

    std::vector<boost::json::value*> find(boost::json::value& json) const;
    std::vector<boost::json::value const*> find(boost::json::value const& json) const;

private:
    std::string _text;
    std::vector<Segment> _segments;
};

enum class JsonConditionOperator
{
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    In,
    Exists
};

// A condition holds if one of the values at the path fulfills the comparison. NotEqual holds if values exist and none of them is equal,
// In expects an array of candidates and Exists a boolean. For Exists and NotEqual, null values count as missing.
struct JsonCondition
{
    JsonPath path;
    JsonConditionOperator op = JsonConditionOperator::Equal;
    boost::json::value value;

    bool isFulfilled(boost::json::value const& json) const;

    static bool areEqual(boost::json::value const& value1, boost::json::value const& value2);
};
