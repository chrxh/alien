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

#include "Exceptions.h"
#include "Vector2D.h"

using std::int32_t;
using std::int64_t;
using std::uint32_t;
using std::uint64_t;

class _FileLogger;
using FileLogger = std::shared_ptr<_FileLogger>;

constexpr float NEAR_ZERO = 1.0e-4f;

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

#define THROW_NOT_IMPLEMENTED() throw std::runtime_error("not implemented")

#define CHECK(expression) \
    if (!(expression)) { \
        throw std::runtime_error("check failed"); \
    }

#define MEMBER_DECLARATION(className, type, name, initialValue) \
    type _##name = initialValue; \
    className& name(type const& name) \
    { \
        _##name = name; \
        return *this; \
    } \
    className& name(type&& name) \
    { \
        _##name = std::move(name); \
        return *this; \
    }

using RealMatrix2D = std::array<std::array<float, 2>, 2>;

struct RealRect
{
    RealVector2D topLeft;
    RealVector2D bottomRight;
};