#include "MathTypes.h"

void IntVector2D::operator-=(IntVector2D const& vec)
{
    x -= vec.x;
    y -= vec.y;
}

void RealVector2D::operator+=(RealVector2D const& vec)
{
    x += vec.x;
    y += vec.y;
}

void RealVector2D::operator-=(RealVector2D const& vec)
{
    x -= vec.x;
    y -= vec.y;
}

RealVector2D RealVector2D::operator+(RealVector2D const& other) const
{
    return RealVector2D{x + other.x, y + other.y};
}

RealVector2D RealVector2D::operator-(RealVector2D const& other) const
{
    return RealVector2D{x - other.x, y - other.y};
}

RealVector2D RealVector2D::operator-() const
{
    return RealVector2D{-x, -y};
}

RealVector2D RealVector2D::operator*(float factor) const
{
    return RealVector2D{x * factor, y * factor};
}

RealVector2D RealVector2D::operator/(float divisor) const
{
    return RealVector2D{x / divisor, y / divisor};
}

uint32_t FloatColorRGB::toRgbColor() const
{
    auto const toInt = [](float value) {
        if (value < 0.0f) {
            value = 0.0f;
        } else if (value > 1.0f) {
            value = 1.0f;
        }
        return static_cast<uint32_t>(value * 255.0f + 0.5f);
    };
    return (toInt(r) << 16) | (toInt(g) << 8) | toInt(b);
}
