#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <charconv>
#include <concepts>
#include <format>

#include <boost/json.hpp>

#include <Base/MathTypes.h>

#include <Data/NeuralNetWeight.h>

#include "SerializationScope.h"
#include "SerializedTypeIds.h"

struct JsonSerializationSettings
{
    bool omitDefaultValues = true;
    bool describeFormat = false;
};

namespace cereal
{
    template <typename T>
    struct IsVariant : std::false_type
    {};
    template <typename... Ts>
    struct IsVariant<std::variant<Ts...>> : std::true_type
    {};

    template <typename T>
    concept JsonPrimitive = std::is_arithmetic_v<T> || std::is_enum_v<T> || std::same_as<T, std::string> || std::same_as<T, RealVector2D>
        || std::same_as<T, IntVector2D> || std::same_as<T, std::chrono::milliseconds> || std::same_as<T, NeuralNetWeight>;

    inline std::string appendJsonPath(std::string const& path, std::string const& name)
    {
        return path.empty() ? name : path + "." + name;
    }

    inline std::invalid_argument createJsonError(std::string const& path, std::string const& message)
    {
        return std::invalid_argument(path.empty() ? message : std::format("'{}': {}", path, message));
    }

    template <typename T>
    boost::json::value toJsonValue(T const& value)
    {
        if constexpr (std::same_as<T, bool>) {
            return value;
        } else if constexpr (std::same_as<T, uint64_t> || std::same_as<T, int64_t>) {
            return boost::json::string(std::to_string(value));
        } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
            return static_cast<int64_t>(value);
        } else if constexpr (std::same_as<T, float>) {
            char buffer[32];
            auto end = std::to_chars(buffer, buffer + sizeof(buffer), value).ptr;
            double result = 0;
            std::from_chars(buffer, end, result);
            return result;
        } else if constexpr (std::is_floating_point_v<T>) {
            return static_cast<double>(value);
        } else if constexpr (std::same_as<T, std::string>) {
            return boost::json::string(value);
        } else if constexpr (std::same_as<T, RealVector2D>) {
            return boost::json::array{toJsonValue(value.x), toJsonValue(value.y)};
        } else if constexpr (std::same_as<T, IntVector2D>) {
            return boost::json::array{value.x, value.y};
        } else if constexpr (std::same_as<T, std::chrono::milliseconds>) {
            return static_cast<int64_t>(value.count());
        } else if constexpr (std::same_as<T, NeuralNetWeight>) {
            return toJsonValue(value.getValue());
        } else if constexpr (IsOptional<T>::value) {
            return value ? toJsonValue(*value) : boost::json::value(nullptr);
        } else if constexpr (IsVector<T>::value) {
            boost::json::array result;
            for (typename T::value_type const& element : value) {
                result.emplace_back(toJsonValue(element));
            }
            return result;
        } else {
            static_assert(sizeof(T) == 0, "Type is not supported by the JSON serialization.");
        }
    }

    template <typename T>
    void fromJsonValue(boost::json::value const& json, T& value, std::string const& path)
    {
        if constexpr (std::same_as<T, bool>) {
            if (!json.is_bool()) {
                throw createJsonError(path, "must be a boolean");
            }
            value = json.as_bool();
        } else if constexpr (std::same_as<T, uint64_t> || std::same_as<T, int64_t>) {
            if (json.is_string()) {
                auto const& text = json.as_string();
                auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
                if (text.empty() || error != std::errc() || end != text.data() + text.size()) {
                    throw createJsonError(path, "must be an integer given as a string of decimal digits");
                }
            } else if (json.is_uint64() && std::in_range<T>(json.as_uint64())) {
                value = static_cast<T>(json.as_uint64());
            } else if (json.is_int64() && std::in_range<T>(json.as_int64())) {
                value = static_cast<T>(json.as_int64());
            } else {
                throw createJsonError(path, "must be an integer given as a string of decimal digits");
            }
        } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
            using Integer = typename std::conditional_t<std::is_enum_v<T>, std::underlying_type<T>, std::type_identity<T>>::type;
            auto invalidInteger = createJsonError(
                path, std::format("must be an integer between {} and {}", std::numeric_limits<Integer>::min() + 0, std::numeric_limits<Integer>::max() + 0));
            int64_t integer = 0;
            if (json.is_int64()) {
                integer = json.as_int64();
            } else if (json.is_uint64() && std::in_range<int64_t>(json.as_uint64())) {
                integer = static_cast<int64_t>(json.as_uint64());
            } else if (json.is_double() && std::trunc(json.as_double()) == json.as_double() && std::abs(json.as_double()) < 1e15) {
                integer = static_cast<int64_t>(json.as_double());
            } else {
                throw invalidInteger;
            }
            if (!std::in_range<Integer>(integer)) {
                throw invalidInteger;
            }
            value = static_cast<T>(integer);
        } else if constexpr (std::is_floating_point_v<T>) {
            if (!json.is_number()) {
                throw createJsonError(path, "must be a number");
            }
            value = static_cast<T>(json.to_number<double>());
        } else if constexpr (std::same_as<T, std::string>) {
            if (!json.is_string()) {
                throw createJsonError(path, "must be a string");
            }
            value = std::string(json.as_string());
        } else if constexpr (std::same_as<T, RealVector2D> || std::same_as<T, IntVector2D>) {
            if (!json.is_array() || json.as_array().size() != 2) {
                throw createJsonError(path, "must be an array [x, y]");
            }
            fromJsonValue(json.as_array().at(0), value.x, path + "[0]");
            fromJsonValue(json.as_array().at(1), value.y, path + "[1]");
        } else if constexpr (std::same_as<T, std::chrono::milliseconds>) {
            int64_t count = 0;
            fromJsonValue(json, count, path);
            value = std::chrono::milliseconds(count);
        } else if constexpr (std::same_as<T, NeuralNetWeight>) {
            float weight = 0;
            fromJsonValue(json, weight, path);
            value = NeuralNetWeight(weight);
        } else if constexpr (IsOptional<T>::value) {
            if (json.is_null()) {
                value.reset();
            } else {
                typename T::value_type element{};
                fromJsonValue(json, element, path);
                value = std::move(element);
            }
        } else if constexpr (IsVector<T>::value) {
            if (!json.is_array()) {
                throw createJsonError(path, "must be an array");
            }
            value.clear();
            for (auto const& [index, jsonElement] : std::views::enumerate(json.as_array())) {
                typename T::value_type element{};
                fromJsonValue(jsonElement, element, std::format("{}[{}]", path, index));
                value.push_back(std::move(element));
            }
        } else {
            static_assert(sizeof(T) == 0, "Type is not supported by the JSON serialization.");
        }
    }

    class JsonArchive
    {
    public:
        JsonArchive(boost::json::object& output, JsonSerializationSettings const& settings, std::string const& path)
            : _output(&output)
            , _settings(settings)
            , _path(path)
        {}

        JsonArchive(boost::json::object const& input, std::string const& path, bool patch)
            : _input(&input)
            , _path(path)
            , _patch(patch)
        {}

        boost::json::object& getOutput() { return *_output; }
        boost::json::object const& getInput() const { return *_input; }
        JsonSerializationSettings const& getSettings() const { return _settings; }
        std::string const& getPath() const { return _path; }
        bool isPatch() const { return _patch; }

        boost::json::value const* consume(std::string const& name)
        {
            _knownNames.insert(name);
            return _input->if_contains(name);
        }

        void checkUnknownNames() const
        {
            for (auto const& [name, value] : *_input) {
                if (!_knownNames.contains(std::string(name))) {
                    std::string knownNames;
                    for (auto const& knownName : _knownNames) {
                        knownNames += (knownNames.empty() ? "" : ", ") + knownName;
                    }
                    throw createJsonError(
                        appendJsonPath(_path, std::string(name)),
                        knownNames.empty() ? "unknown field, no fields are allowed here" : std::format("unknown field, allowed are: {}", knownNames));
                }
            }
        }

    private:
        boost::json::object* _output = nullptr;
        boost::json::object const* _input = nullptr;
        JsonSerializationSettings _settings;
        std::string _path;
        bool _patch = false;
        std::set<std::string> _knownNames;
    };

    template <typename T>
    boost::json::value saveDescToJson(T const& value, JsonSerializationSettings const& settings, std::string const& path);

    template <typename T>
    void loadDescFromJson(boost::json::value const& json, T& value, std::string const& path, bool patch);

    class JsonSerializationScope
    {
    public:
        JsonSerializationScope(SerializationTask task, JsonArchive& ar)
            : _task(task)
            , _ar(ar)
        {}

        template <typename T>
        void addMember(SerializationKey key, T& value, T const& defaultValue)
        {
            auto name = getName(key);
            if (_task == SerializationTask::Save) {
                auto const& settings = _ar.getSettings();
                if constexpr (std::equality_comparable<T>) {
                    if (settings.omitDefaultValues && !settings.describeFormat && value == defaultValue) {
                        return;
                    }
                }
                insert(name, toJsonValue(value));
            } else if (auto json = _ar.consume(name)) {
                fromJsonValue(*json, value, appendJsonPath(_ar.getPath(), name));
            } else if (!_ar.isPatch()) {
                value = defaultValue;
            }
        }

        template <typename T>
        void addFixedSizeMember(SerializationKey key, std::vector<T>& value, std::vector<T> const& defaultValue)
        {
            addMember(key, value, defaultValue);
            if (_task == SerializationTask::Load && value.size() != defaultValue.size()) {
                throw createJsonError(appendJsonPath(_ar.getPath(), getName(key)), std::format("must have exactly {} entries", defaultValue.size()));
            }
        }

        template <typename T>
        void addDesc(SerializationKey key, T& value)
        {
            auto name = getName(key);
            auto path = appendJsonPath(_ar.getPath(), name);
            if (_task == SerializationTask::Save) {
                auto const& settings = _ar.getSettings();
                if (settings.omitDefaultValues && !settings.describeFormat) {
                    if constexpr (IsOptional<T>::value) {
                        if (!value) {
                            return;
                        }
                    } else if constexpr (IsVector<T>::value) {
                        if (value.empty()) {
                            return;
                        }
                    }
                }
                insert(name, saveDescToJson(value, settings, path));
            } else if (auto json = _ar.consume(name)) {
                loadDescFromJson(*json, value, path, _ar.isPatch());
            }
        }

    private:
        std::string getName(SerializationKey key) const
        {
            if (!key.name) {
                throw std::logic_error(std::format("The serialization key {} has no JSON name.", key.id));
            }
            return key.name;
        }

        void insert(std::string const& name, boost::json::value&& value)
        {
            auto [iter, inserted] = _ar.getOutput().emplace(name, std::move(value));
            if (!inserted) {
                throw std::logic_error(std::format("The JSON name '{}' is used twice.", appendJsonPath(_ar.getPath(), name)));
            }
        }

        SerializationTask _task;
        JsonArchive& _ar;
    };

    inline JsonSerializationScope getSerializationScope(SerializationTask task, JsonArchive& ar)
    {
        return JsonSerializationScope(task, ar);
    }

    template <typename Variant>
    std::string getVariantAlternativeNames()
    {
        std::string result;
        [&]<size_t... Indices>(std::index_sequence<Indices...>) {
            ((result += (result.empty() ? "" : ", ") + std::string(SerializedTypeId<std::variant_alternative_t<Indices, Variant>>::name)), ...);
        }(std::make_index_sequence<std::variant_size_v<Variant>>());
        return result;
    }

    template <typename T>
    boost::json::value saveDescToJson(T const& value, JsonSerializationSettings const& settings, std::string const& path)
    {
        if constexpr (IsVector<T>::value) {
            boost::json::array result;
            if (settings.describeFormat && value.empty()) {
                result.emplace_back(saveDescToJson(typename T::value_type(), settings, path + "[0]"));
            }
            for (auto const& [index, element] : std::views::enumerate(value)) {
                result.emplace_back(saveDescToJson(element, settings, std::format("{}[{}]", path, index)));
            }
            return result;
        } else if constexpr (IsOptional<T>::value) {
            if (settings.describeFormat) {
                return saveDescToJson(value.value_or(typename T::value_type()), settings, path);
            }
            return value ? saveDescToJson(*value, settings, path) : boost::json::value(nullptr);
        } else if constexpr (IsVariant<T>::value) {
            boost::json::object result;
            if (settings.describeFormat) {
                boost::json::object alternatives;
                [&]<size_t... Indices>(std::index_sequence<Indices...>) {
                    ((alternatives[SerializedTypeId<std::variant_alternative_t<Indices, T>>::name] =
                          saveDescToJson(std::variant_alternative_t<Indices, T>(), settings, path)),
                     ...);
                }(std::make_index_sequence<std::variant_size_v<T>>());
                result["oneOf"] = std::move(alternatives);
            } else {
                std::visit(
                    [&](auto const& alternative) {
                        using Alternative = std::decay_t<decltype(alternative)>;
                        result[SerializedTypeId<Alternative>::name] = saveDescToJson(alternative, settings, path);
                    },
                    value);
            }
            return result;
        } else if constexpr (JsonPrimitive<T>) {
            return toJsonValue(value);
        } else {
            boost::json::object result;
            JsonArchive archive(result, settings, path);
            loadSave(SerializationTask::Save, archive, const_cast<T&>(value));
            return result;
        }
    }

    template <typename T>
    void loadDescFromJson(boost::json::value const& json, T& value, std::string const& path, bool patch)
    {
        if constexpr (IsVector<T>::value) {
            if (!json.is_array()) {
                throw createJsonError(path, "must be an array");
            }
            T result;
            for (auto const& [index, jsonElement] : std::views::enumerate(json.as_array())) {
                auto element = patch && std::cmp_less(index, value.size()) ? value.at(index) : typename T::value_type();
                loadDescFromJson(jsonElement, element, std::format("{}[{}]", path, index), patch);
                result.push_back(std::move(element));
            }
            value = std::move(result);
        } else if constexpr (IsOptional<T>::value) {
            if (json.is_null()) {
                value.reset();
            } else {
                auto element = patch && value ? *value : typename T::value_type();
                loadDescFromJson(json, element, path, patch);
                value = std::move(element);
            }
        } else if constexpr (IsVariant<T>::value) {
            auto alternativeNames = getVariantAlternativeNames<T>();
            if (!json.is_object() || json.as_object().size() != 1) {
                throw createJsonError(path, std::format("must be an object with exactly one of the fields {}", alternativeNames));
            }
            auto const& [name, jsonAlternative] = *json.as_object().begin();
            auto found = false;
            [&]<size_t... Indices>(std::index_sequence<Indices...>) {
                auto loadIfMatching = [&]<size_t Index>() {
                    if (name != SerializedTypeId<std::variant_alternative_t<Index, T>>::name) {
                        return false;
                    }
                    auto& alternative = patch && value.index() == Index ? std::get<Index>(value) : value.template emplace<Index>();
                    loadDescFromJson(jsonAlternative, alternative, appendJsonPath(path, std::string(name)), patch);
                    return true;
                };
                found = (loadIfMatching.template operator()<Indices>() || ...);
            }(std::make_index_sequence<std::variant_size_v<T>>());
            if (!found) {
                throw createJsonError(appendJsonPath(path, std::string(name)), std::format("unknown alternative, allowed are: {}", alternativeNames));
            }
        } else if constexpr (JsonPrimitive<T>) {
            fromJsonValue(json, value, path);
        } else {
            if (!json.is_object()) {
                throw createJsonError(path, "must be an object");
            }
            JsonArchive archive(json.as_object(), path, patch);
            loadSave(SerializationTask::Load, archive, value);
            archive.checkUnknownNames();
        }
    }
}
