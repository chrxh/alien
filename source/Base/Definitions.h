#pragma once

#include <array>
#include <map>
#include <set>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <memory>
#include <initializer_list>
#include <cstdint>

#include "Exceptions.h"
#include "Vector2D.h"
#include "Macros.h"

using std::int32_t;
using std::int64_t;
using std::uint32_t;
using std::uint64_t;

class _FileLogger;
using FileLogger = std::shared_ptr<_FileLogger>;

inline constexpr float NEAR_ZERO = 1.0e-4f;

template <typename T>
inline float toFloat(T const& value)
{
    return static_cast<float>(value);
}

template <typename T>
inline double toDouble(T const& value)
{
    return static_cast<double>(value);
}

template<typename T>
inline int toInt(T const& value)
{
    return static_cast<int>(value);
}

template <typename T>
inline uint8_t toUInt8(T const& value)
{
    return static_cast<uint8_t>(value);
}

template <typename T>
inline uint16_t toUInt16(T const& value)
{
    return static_cast<uint16_t>(value);
}

template <typename T>
inline uint32_t toUInt32(T const& value)
{
    return static_cast<uint32_t>(value);
}

