#pragma once

#include <functional>
#include <initializer_list>

struct IntVector2D
{
    int x = 0;
    int y = 0;

    IntVector2D() = default;
    IntVector2D(std::initializer_list<int> l);
    bool operator==(IntVector2D const& vec) const;
    void operator-=(IntVector2D const& vec);
};

namespace std
{
    template <>
    struct hash<IntVector2D>
    {
        std::size_t operator()(IntVector2D const& value) const
        {
            using std::hash;
            using std::size_t;

            size_t res = 17;
            res = res * 31 + hash<int>()(value.x);
            res = res * 31 + hash<int>()(value.y);
            return res;
        }
    };

}

struct RealVector2D
{
    float x = 0.0f;
    float y = 0.0f;

    RealVector2D() = default;
    RealVector2D(float x_, float y_);
    RealVector2D(std::initializer_list<float> l);
    bool operator==(RealVector2D const& other) const;
    void operator+=(RealVector2D const& vec);
    void operator-=(RealVector2D const& vec);
    template <typename T>
    void operator*=(T factor)
    {
        x *= factor;
        y *= factor;
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
    RealVector2D operator-() const;
    RealVector2D operator*(float factor) const;
    RealVector2D operator/(float divisor) const;
};

inline IntVector2D toIntVector2D(RealVector2D const& v)
{
    return {static_cast<int>(v.x), static_cast<int>(v.y)};
}
