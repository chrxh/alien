#include "JsonPath.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <charconv>
#include <format>
#include <span>

#include <boost/json.hpp>

#include <Base/StringHelper.h>

JsonPath::JsonPath(std::string const& text)
    : _text(text)
{
    auto invalidPath = [&](std::string const& reason) { return std::invalid_argument(std::format("Invalid path '{}': {}.", text, reason)); };
    if (text.empty()) {
        throw invalidPath("the path is empty");
    }
    size_t pos = 0;
    while (pos < text.size()) {
        if (text.at(pos) == '[') {
            auto end = text.find(']', pos);
            if (end == std::string::npos) {
                throw invalidPath("'[' without ']'");
            }
            auto content = text.substr(pos + 1, end - pos - 1);
            if (content == "*") {
                _segments.emplace_back(Segment{.type = Segment::Type::AnyIndex});
            } else {
                int index = 0;
                auto [contentEnd, error] = std::from_chars(content.data(), content.data() + content.size(), index);
                if (content.empty() || error != std::errc() || contentEnd != content.data() + content.size() || index < 0) {
                    throw invalidPath(std::format("'[{}]' must contain a non-negative index or '*'", content));
                }
                _segments.emplace_back(Segment{.type = Segment::Type::Index, .index = index});
            }
            pos = end + 1;
            continue;
        }
        if (!_segments.empty()) {
            if (text.at(pos) != '.') {
                throw invalidPath(std::format("unexpected '{}' at position {}", text.at(pos), pos));
            }
            ++pos;
        }
        auto end = std::min(text.find_first_of(".[]", pos), text.size());
        auto key = text.substr(pos, end - pos);
        if (key.empty()) {
            throw invalidPath("empty field name");
        }
        _segments.emplace_back(key == "*" ? Segment{.type = Segment::Type::AnyKey} : Segment{.type = Segment::Type::Key, .key = key});
        pos = end;
    }
}

namespace
{
    boost::json::object const* getAlternatives(boost::json::value const& format)
    {
        if (auto object = format.if_object()) {
            if (auto alternatives = object->if_contains("oneOf")) {
                return alternatives->if_object();
            }
        }
        return nullptr;
    }

    std::string appendSegment(std::string const& path, JsonPath::Segment const& segment)
    {
        switch (segment.type) {
        case JsonPath::Segment::Type::Key:
            return path.empty() ? segment.key : path + "." + segment.key;
        case JsonPath::Segment::Type::Index:
            return std::format("{}[{}]", path, segment.index);
        case JsonPath::Segment::Type::AnyKey:
            return path.empty() ? "*" : path + ".*";
        case JsonPath::Segment::Type::AnyIndex:
            return path + "[*]";
        }
        return path;
    }

    std::invalid_argument
    createUnknownPathError(std::string const& pathText, JsonPath::Segment const& segment, boost::json::value const& parentFormat, std::string const& parentPath)
    {
        auto parentName = parentPath.empty() ? std::string("the top level") : std::format("'{}'", parentPath);
        auto prefix = std::format("Unknown path '{}': ", pathText);
        if (segment.type == JsonPath::Segment::Type::Index || segment.type == JsonPath::Segment::Type::AnyIndex) {
            return std::invalid_argument(prefix + std::format("{} is not an array.", parentName));
        }
        auto alternatives = getAlternatives(parentFormat);
        auto fields = alternatives ? alternatives : parentFormat.if_object();
        if (!fields || fields->empty()) {
            return std::invalid_argument(prefix + std::format("{} has no fields.", parentName));
        }
        std::vector<std::string> names;
        for (auto const& [name, value] : *fields) {
            names.emplace_back(name);
        }
        return std::invalid_argument(
            prefix
            + std::format(
                "'{}' is not {} of {}, allowed are: {}.",
                segment.key,
                alternatives ? "an alternative" : "a field",
                parentName,
                StringHelper::formatEnumeration(names)));
    }
}

std::vector<boost::json::value const*> JsonPath::resolveFormat(boost::json::value const& format) const
{
    std::vector<boost::json::value const*> result{&format};
    std::string resolvedPath;
    for (auto const& segment : _segments) {
        std::vector<boost::json::value const*> children;
        for (auto value : result) {
            auto alternatives = getAlternatives(*value);
            auto fields = alternatives ? alternatives : value->if_object();
            switch (segment.type) {
            case Segment::Type::Key:
                if (fields) {
                    if (auto child = fields->if_contains(segment.key)) {
                        children.emplace_back(child);
                    }
                }
                break;
            case Segment::Type::AnyKey:
                if (fields) {
                    for (auto const& [name, child] : *fields) {
                        children.emplace_back(&child);
                    }
                }
                break;
            case Segment::Type::Index:
            case Segment::Type::AnyIndex:
                if (auto array = value->if_array(); array && !array->empty()) {
                    children.emplace_back(&array->front());
                }
                break;
            }
        }
        if (children.empty()) {
            throw createUnknownPathError(_text, segment, *result.front(), resolvedPath);
        }
        result = std::move(children);
        resolvedPath = appendSegment(resolvedPath, segment);
    }
    return result;
}

namespace
{
    template <typename Value>
    void findValues(Value& json, std::span<JsonPath::Segment const> segments, std::vector<Value*>& result)
    {
        if (segments.empty()) {
            result.emplace_back(&json);
            return;
        }
        auto const& segment = segments.front();
        auto remainingSegments = segments.subspan(1);
        switch (segment.type) {
        case JsonPath::Segment::Type::Key:
            if (auto object = json.if_object()) {
                if (auto child = object->if_contains(segment.key)) {
                    findValues(*child, remainingSegments, result);
                }
            }
            break;
        case JsonPath::Segment::Type::AnyKey:
            if (auto object = json.if_object()) {
                for (auto& [name, child] : *object) {
                    findValues(child, remainingSegments, result);
                }
            }
            break;
        case JsonPath::Segment::Type::Index:
            if (auto array = json.if_array(); array && std::cmp_less(segment.index, array->size())) {
                findValues(array->at(segment.index), remainingSegments, result);
            }
            break;
        case JsonPath::Segment::Type::AnyIndex:
            if (auto array = json.if_array()) {
                for (auto& element : *array) {
                    findValues(element, remainingSegments, result);
                }
            }
            break;
        }
    }
}

std::vector<boost::json::value*> JsonPath::find(boost::json::value& json) const
{
    std::vector<boost::json::value*> result;
    findValues(json, std::span(_segments), result);
    return result;
}

std::vector<boost::json::value const*> JsonPath::find(boost::json::value const& json) const
{
    std::vector<boost::json::value const*> result;
    findValues(json, std::span(_segments), result);
    return result;
}

bool JsonCondition::isFulfilled(boost::json::value const& json) const
{
    auto values = path.find(json);
    if (op == JsonConditionOperator::Exists || op == JsonConditionOperator::NotEqual) {
        std::erase_if(values, [](auto const* candidate) { return candidate->is_null(); });
    }
    auto isAnyValue = [&](auto const& predicate) { return std::ranges::any_of(values, [&](auto const* candidate) { return predicate(*candidate); }); };
    auto isAnyNumber = [&](auto const& comparison) {
        return isAnyValue([&](boost::json::value const& candidate) {
            return candidate.is_number() && value.is_number() && comparison(candidate.to_number<double>(), value.to_number<double>());
        });
    };
    switch (op) {
    case JsonConditionOperator::Equal:
        return isAnyValue([&](boost::json::value const& candidate) { return areEqual(candidate, value); });
    case JsonConditionOperator::NotEqual:
        return !values.empty() && !isAnyValue([&](boost::json::value const& candidate) { return areEqual(candidate, value); });
    case JsonConditionOperator::Less:
        return isAnyNumber(std::less<double>());
    case JsonConditionOperator::LessEqual:
        return isAnyNumber(std::less_equal<double>());
    case JsonConditionOperator::Greater:
        return isAnyNumber(std::greater<double>());
    case JsonConditionOperator::GreaterEqual:
        return isAnyNumber(std::greater_equal<double>());
    case JsonConditionOperator::In:
        return value.is_array() && isAnyValue([&](boost::json::value const& candidate) {
                   return std::ranges::any_of(value.as_array(), [&](boost::json::value const& element) { return areEqual(candidate, element); });
               });
    case JsonConditionOperator::Exists:
        return values.empty() != (!value.is_bool() || value.as_bool());
    }
    return false;
}

bool JsonCondition::areEqual(boost::json::value const& value1, boost::json::value const& value2)
{
    if (value1.is_number() && value2.is_number()) {
        return value1.to_number<double>() == value2.to_number<double>();
    }
    return value1 == value2;
}
