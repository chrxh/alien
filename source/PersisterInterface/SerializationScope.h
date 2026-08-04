#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <Base/MathTypes.h>

#include <EngineInterface/NeuralNetWeight.h>

#include "SerializedTypeIds.h"

#define SPLIT_SERIALIZATION(Classname) \
    template <class Archive> \
    void save(Archive& ar, Classname const& data) \
    { \
        loadSave(SerializationTask::Save, ar, const_cast<Classname&>(data)); \
    } \
    template <class Archive> \
    void load(Archive& ar, Classname& data) \
    { \
        loadSave(SerializationTask::Load, ar, data); \
    }

enum class SerializationTask
{
    Load,
    Save
};

// Deferred read operations are executed in a destructor, which must not throw: failures are collected here instead
inline thread_local bool deserializationFailed = false;

// Provides the context for one deserialization
class DeserializationContext
{
public:
    DeserializationContext() { deserializationFailed = false; }

    DeserializationContext(DeserializationContext const&) = delete;
    DeserializationContext& operator=(DeserializationContext const&) = delete;

    void throwOnFailure() const
    {
        if (deserializationFailed) {
            throw std::runtime_error("The file could not be read.");
        }
    }
};

namespace cereal
{
    using VariantData = std::variant<
        int,
        float,
        bool,
        double,
        std::string,
        uint64_t,
        uint32_t,
        uint16_t,
        uint8_t,
        int64_t,
        int16_t,
        int8_t,
        RealVector2D,
        std::optional<bool>,
        std::optional<uint64_t>,
        std::optional<uint8_t>,
        std::optional<int8_t>,
        std::optional<int>,
        std::optional<float>,
        std::optional<RealVector2D>,
        std::vector<bool>,
        std::vector<uint8_t>,
        std::vector<int8_t>,
        std::vector<int>,
        std::vector<float>,
        std::vector<RealVector2D>,
        std::vector<std::vector<uint8_t>>,
        std::vector<std::vector<int8_t>>,
        std::vector<std::vector<int>>,
        std::vector<std::vector<float>>,
        IntVector2D,
        std::chrono::milliseconds>;

    using AttributeMap = std::unordered_map<int, VariantData>;

    /************************************************************************/
    /* Type conversion for attribute values                                 */
    /************************************************************************/

    template <typename T>
    struct IsOptional : std::false_type
    {};
    template <typename T>
    struct IsOptional<std::optional<T>> : std::true_type
    {};

    template <typename T>
    struct IsVector : std::false_type
    {};
    template <typename T>
    struct IsVector<std::vector<T>> : std::true_type
    {};

    template <typename T>
    struct IsScalar : std::bool_constant<std::is_arithmetic_v<T> || std::is_enum_v<T>>
    {};

    // Converts a value read from a file to the type the corresponding member has today. This keeps files
    // readable when the type of a member changes (e.g. int -> uint8_t). Returns false if there is no conversion.
    template <typename From, typename To>
    bool convertValue(From const& from, To& to)
    {
        if constexpr (std::is_same_v<From, To>) {
            to = from;
            return true;
        } else if constexpr (IsOptional<From>::value && IsOptional<To>::value) {
            if (!from.has_value()) {
                to.reset();
                return true;
            }
            typename To::value_type converted{};
            if (!convertValue(*from, converted)) {
                return false;
            }
            to = std::move(converted);
            return true;
        } else if constexpr (IsOptional<From>::value) {
            return from.has_value() && convertValue(*from, to);
        } else if constexpr (IsOptional<To>::value) {
            typename To::value_type converted{};
            if (!convertValue(from, converted)) {
                return false;
            }
            to = std::move(converted);
            return true;
        } else if constexpr (IsVector<From>::value && IsVector<To>::value) {
            to.clear();
            to.reserve(from.size());
            for (auto const& element : from) {
                typename To::value_type converted{};
                if (!convertValue(element, converted)) {
                    return false;
                }
                to.push_back(std::move(converted));
            }
            return true;
        } else if constexpr (IsScalar<From>::value && IsScalar<To>::value) {
            to = static_cast<To>(from);
            return true;
        } else {
            return false;
        }
    }

    template <typename T>
    bool convertVariantData(VariantData const& variantData, T& result)
    {
        return std::visit([&result](auto const& value) { return convertValue(value, result); }, variantData);
    }

    /************************************************************************/
    /* Serialization scope                                                  */
    /************************************************************************/

    // RAII pattern
    template <class Archive>
    class SerializationScope
    {
    public:
        SerializationScope(SerializationTask task, Archive& ar)
            : _task(task)
            , _ar(ar)
        {
            if (_task == SerializationTask::Load) {
                _ar(_attributeMap);
            }
        }

        ~SerializationScope()
        {
            if (_task == SerializationTask::Load && deserializationFailed) {

                // Reading further would only operate on garbage
                return;
            }
            try {
                processDeferredDescOps();
            } catch (...) {

                // A destructor must not throw: the failure is reported by the enclosing DeserializationContext
                deserializationFailed = true;
            }
        }

        SerializationScope(const SerializationScope&) = delete;
        SerializationScope& operator=(const SerializationScope&) = delete;

        SerializationScope(SerializationScope&&) = default;
        SerializationScope& operator=(SerializationScope&&) = default;

        // Implicit conversion to reference
        operator std::unordered_map<int, VariantData>&() & { return _attributeMap; }

        template <typename T>
        void addMember(int key, T& value, T const& defaultValue)
        {
            if (_task == SerializationTask::Load) {
                auto findResult = _attributeMap.find(key);
                if (findResult == _attributeMap.end() || !convertVariantData(findResult->second, value)) {
                    value = defaultValue;
                }
            } else {
                _attributeMap.emplace(key, value);
            }
        }

        template <typename T>
        void addDesc(int key, T& value)
        {
            if (_task == SerializationTask::Save) {
                // Defer the save operation
                addDeferredDescOp(key, [this, &value]() {
                    // Serialize to buffer
                    std::ostringstream ss(std::ios::binary);
                    {
                        cereal::PortableBinaryOutputArchive bufferAr(ss);
                        bufferAr(value);
                    }
                    auto serializedData = std::move(ss).str();
                    uint64_t dataSize = serializedData.size();

                    // Write size-prefixed data
                    _ar(dataSize);
                    _ar(cereal::binary_data(serializedData.data(), dataSize));
                });
            } else {
                // Defer the load operation
                addDeferredDescOp(key, [this, &value]() {
                    // Read size-prefixed data
                    uint64_t dataSize = 0;
                    _ar(dataSize);

                    // Read serialized data into buffer
                    std::string serializedData(dataSize, '\0');
                    _ar(cereal::binary_data(serializedData.data(), dataSize));

                    // Deserialize from buffer
                    std::istringstream ss(std::move(serializedData), std::ios::binary);
                    cereal::PortableBinaryInputArchive bufferAr(ss);
                    bufferAr(value);
                });
            }
        }

        // For vectors whose size is fixed: files saved with a different layout may contain differently
        // sized vectors, so the loaded values are merged into a default-sized vector
        template <typename T>
        void addFixedSizeMember(int key, std::vector<T>& value, std::vector<T> const& defaultValue)
        {
            addMember(key, value, defaultValue);
            if (_task == SerializationTask::Load && value.size() != defaultValue.size()) {
                auto adaptedValue = defaultValue;
                std::copy_n(value.begin(), std::min(value.size(), adaptedValue.size()), adaptedValue.begin());
                value = std::move(adaptedValue);
            }
        }

        // Specialized overload for std::vector<NeuralNetWeight> - converts to/from std::vector<int8_t> for serialization
        void addMember(int key, std::vector<NeuralNetWeight>& value, std::vector<NeuralNetWeight> const& defaultValue)
        {
            if (_task == SerializationTask::Load) {
                std::vector<int8_t> rawValues;
                auto findResult = _attributeMap.find(key);
                if (findResult == _attributeMap.end() || !convertVariantData(findResult->second, rawValues)) {
                    value = defaultValue;
                    return;
                }
                value.resize(rawValues.size());
                for (auto const& [weight, rawValue] : std::views::zip(value, rawValues)) {
                    weight = NeuralNetWeight::fromRawValue(static_cast<uint8_t>(rawValue));
                }
            } else {
                std::vector<int8_t> rawValues;
                rawValues.reserve(value.size());
                for (auto const& weight : value) {
                    rawValues.push_back(weight.rawValue);
                }
                _attributeMap.emplace(key, rawValues);
            }
        }

    private:
        void processDeferredDescOps()
        {
            std::sort(_deferredDescOps.begin(), _deferredDescOps.end(), [](auto const& left, auto const& right) { return left.id < right.id; });

            // Process deferred operations
            if (_task == SerializationTask::Save) {

                // Save map first
                _ar(_attributeMap);

                // Save sorted ids
                std::vector<int> sortedIds;
                sortedIds.reserve(_deferredDescOps.size());
                for (const auto& op : _deferredDescOps) {
                    sortedIds.push_back(op.id);
                }
                _ar(sortedIds);

                // Then write size-prefixed ContentDesc data in sorted id order
                for (auto const& op : _deferredDescOps) {
                    op.serializeFunc();
                }
            } else {

                // Read sorted ids
                std::vector<int> savedIds;
                _ar(savedIds);

                // For each id, check if we have a deferred read operation, otherwise skip bytes
                auto deferredOpIndex = 0;
                auto deferredOpSize = _deferredDescOps.size();
                for (int savedId : savedIds) {
                    // deferredOpIndex is an optimization to avoid
                    // `std::find_if(_deferredDescOps.begin(), _deferredDescOps.end(), [id](const auto& op) { return op.id == id; });`
                    // for each savedId (savedIds and _deferredDescOps are sorted)
                    while (deferredOpIndex < deferredOpSize && _deferredDescOps.at(deferredOpIndex).id < savedId) {
                        ++deferredOpIndex;
                    }

                    if (deferredOpIndex < deferredOpSize && _deferredDescOps.at(deferredOpIndex).id == savedId) {
                        // We want to read this ContentDesc - execute the read
                        _deferredDescOps.at(deferredOpIndex).serializeFunc();
                    } else {
                        // Skip this ContentDesc - read size and skip data
                        uint64_t dataSize = 0;
                        _ar(dataSize);
                        std::vector<uint8_t> buffer(dataSize);
                        _ar(cereal::binary_data(buffer.data(), dataSize));
                    }
                }
            }
        }

        void addDeferredDescOp(int id, std::function<void()> serializeFunc) { _deferredDescOps.push_back({id, std::move(serializeFunc)}); }

        struct DeferredOperation
        {
            int id;
            std::function<void()> serializeFunc;
        };

        SerializationTask _task;
        Archive& _ar;
        AttributeMap _attributeMap;
        std::vector<DeferredOperation> _deferredDescOps;
    };

    template <class Archive>
    SerializationScope<Archive> getSerializationScope(SerializationTask task, Archive& ar)
    {
        return SerializationScope<Archive>(task, ar);
    }

    template <class Archive>
    void serialize(Archive& ar, IntVector2D& data)
    {
        ar(data.x, data.y);
    }
    template <class Archive>
    void serialize(Archive& ar, RealVector2D& data)
    {
        ar(data.x, data.y);
    }

    template <class Archive>
    void loadSave(SerializationTask task, Archive& ar, std::chrono::milliseconds& data)
    {
        auto count = static_cast<uint64_t>(data.count());
        ar(count);
        if (task == SerializationTask::Load) {
            data = std::chrono::milliseconds(count);
        }
    }
    SPLIT_SERIALIZATION(std::chrono::milliseconds)
}
