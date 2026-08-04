#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string>
#include <variant>
#include <vector>

#include <cereal/cereal.hpp>

#include <Base/MathTypes.h>

namespace cereal
{
    // A std::variant is written as a type id followed by the value. The type id is part of the file format:
    // the alternatives of a variant may be reordered or removed at any time, but an id must never change its
    // meaning and must never be reused for another type. New alternatives get the next free id of their variant.
    // An unregistered type does not compile. The ids of the description types are registered in SerializerService.cpp.
    template <typename T>
    struct SerializedTypeId;

#define REGISTER_SERIALIZED_TYPE(Type, Id) \
    template <> \
    struct SerializedTypeId<Type> \
    { \
        static constexpr int32_t value = Id; \
    };

    // Attribute values
    REGISTER_SERIALIZED_TYPE(int, 0)
    REGISTER_SERIALIZED_TYPE(float, 1)
    REGISTER_SERIALIZED_TYPE(bool, 2)
    REGISTER_SERIALIZED_TYPE(double, 3)
    REGISTER_SERIALIZED_TYPE(std::string, 4)
    REGISTER_SERIALIZED_TYPE(uint64_t, 5)
    REGISTER_SERIALIZED_TYPE(uint32_t, 6)
    REGISTER_SERIALIZED_TYPE(uint16_t, 7)
    REGISTER_SERIALIZED_TYPE(uint8_t, 8)
    REGISTER_SERIALIZED_TYPE(int64_t, 9)
    REGISTER_SERIALIZED_TYPE(int16_t, 10)
    REGISTER_SERIALIZED_TYPE(int8_t, 11)
    REGISTER_SERIALIZED_TYPE(RealVector2D, 12)
    REGISTER_SERIALIZED_TYPE(std::optional<bool>, 13)
    REGISTER_SERIALIZED_TYPE(std::optional<uint64_t>, 14)
    REGISTER_SERIALIZED_TYPE(std::optional<uint8_t>, 15)
    REGISTER_SERIALIZED_TYPE(std::optional<int8_t>, 16)
    REGISTER_SERIALIZED_TYPE(std::optional<int>, 17)
    REGISTER_SERIALIZED_TYPE(std::optional<float>, 18)
    REGISTER_SERIALIZED_TYPE(std::optional<RealVector2D>, 19)
    REGISTER_SERIALIZED_TYPE(std::vector<bool>, 20)
    REGISTER_SERIALIZED_TYPE(std::vector<uint8_t>, 21)
    REGISTER_SERIALIZED_TYPE(std::vector<int8_t>, 22)
    REGISTER_SERIALIZED_TYPE(std::vector<int>, 23)
    REGISTER_SERIALIZED_TYPE(std::vector<float>, 24)
    REGISTER_SERIALIZED_TYPE(std::vector<RealVector2D>, 25)
    REGISTER_SERIALIZED_TYPE(std::vector<std::vector<uint8_t>>, 26)
    REGISTER_SERIALIZED_TYPE(std::vector<std::vector<int8_t>>, 27)
    REGISTER_SERIALIZED_TYPE(std::vector<std::vector<int>>, 28)
    REGISTER_SERIALIZED_TYPE(std::vector<std::vector<float>>, 29)
    REGISTER_SERIALIZED_TYPE(IntVector2D, 30)
    REGISTER_SERIALIZED_TYPE(std::chrono::milliseconds, 31)


    template <typename... Ts>
    constexpr bool hasUniqueSerializedTypeIds(std::variant<Ts...> const*)
    {
        std::array typeIds = {SerializedTypeId<Ts>::value...};
        std::ranges::sort(typeIds);
        return std::ranges::adjacent_find(typeIds) == typeIds.end();
    }

    template <class Archive, typename Variant, size_t... Indices>
    bool loadVariantAlternative(Archive& ar, int32_t typeId, Variant& data, std::index_sequence<Indices...>)
    {
        auto loadIfMatching = [&]<size_t Index>() {
            using Alternative = std::variant_alternative_t<Index, Variant>;
            if (SerializedTypeId<Alternative>::value != typeId) {
                return false;
            }
            data.template emplace<Index>();
            ar(std::get<Index>(data));
            return true;
        };
        return (loadIfMatching.template operator()<Indices>() || ...);
    }

    template <class Archive, typename... Ts>
    void save(Archive& ar, std::variant<Ts...> const& data)
    {
        static_assert(hasUniqueSerializedTypeIds(static_cast<std::variant<Ts...> const*>(nullptr)), "Serialized type ids must be unique.");

        auto typeId = std::visit([](auto const& value) { return SerializedTypeId<std::decay_t<decltype(value)>>::value; }, data);
        ar(typeId);
        std::visit([&ar](auto const& value) { ar(value); }, data);
    }

    template <class Archive, typename... Ts>
    void load(Archive& ar, std::variant<Ts...>& data)
    {
        static_assert(hasUniqueSerializedTypeIds(static_cast<std::variant<Ts...> const*>(nullptr)), "Serialized type ids must be unique.");

        int32_t typeId = 0;
        ar(typeId);
        if (!loadVariantAlternative(ar, typeId, data, std::index_sequence_for<Ts...>())) {
            throw Exception("Unknown type id when deserializing std::variant.");
        }
    }
}
