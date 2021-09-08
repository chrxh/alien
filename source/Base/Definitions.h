#pragma once

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

#include "DllExport.h"
#include "Exceptions.h"

using boost::optional;
using boost::shared_ptr;
using std::int32_t;
using std::int64_t;
using std::list;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::uint32_t;
using std::uint64_t;
using std::unordered_map;
using std::unordered_set;
using std::vector;

const double FLOATINGPOINT_HIGH_PRECISION = 1.0e-7;
const double FLOATINGPOINT_MEDIUM_PRECISION = 1.0e-4;
const double FLOATINGPOINT_LOW_PRECISION = 1.0e-1;

template <typename T>
inline float toFloat(T const& value)
{
    return static_cast<float>(value);
}

template<typename T>
inline int toInt(T const& value)
{
    return static_cast<int>(value);
}

#define THROW_NOT_IMPLEMENTED() throw std::runtime_error("not implemented")

#define CHECK(expression) \
    if (!(expression)) { \
        throw BugReportException("check failed"); \
    }

#define MEMBER_DECLARATION(className, type, name, initialValue) \
    type _##name = initialValue; \
    className& name(type const& name) \
    { \
        _##name = name; \
        return *this; \
    }

struct BASE_EXPORT IntVector2D
{
    int x = 0;
    int y = 0;

    IntVector2D() = default;
    IntVector2D(std::initializer_list<int> l);
    bool operator==(IntVector2D const& vec) const;
    void operator-=(IntVector2D const& vec);
};

struct BASE_EXPORT RealVector2D
{
    float x = 0.0f;
    float y = 0.0f;

    RealVector2D() = default;
    RealVector2D(float x_, float y_);
    RealVector2D(std::initializer_list<float> l);
    bool operator==(RealVector2D const& vec) const;
    bool operator!=(RealVector2D const& vec) const { return !operator==(vec); }
    void operator+=(RealVector2D const& vec);
    void operator-=(RealVector2D const& vec);
    template <typename T>
    void operator*=(T divisor)
    {
        x *= divisor;
        y *= divisor;
    }
    template <typename T>
    void operator/=(T divisor)
    {
        x /= divisor;
        y /= divisor;
    }
    template <typename T>
    RealVector2D operator*(T factor) const
    {
        return RealVector2D{x * factor, y * factor};
    }
    RealVector2D operator+(RealVector2D const& other) const;
    RealVector2D operator-(RealVector2D const& other) const;
    RealVector2D operator/(float divisor) const;
};
