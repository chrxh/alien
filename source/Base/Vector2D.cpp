#include "Vector2D.h"

RealVector2D::RealVector2D(float x_, float y_)
    : x(x_)
    , y(y_)
{}

RealVector2D::RealVector2D(std::initializer_list<float> l)
{
    auto it = l.begin();
    x = *it++;
    y = *it;
}

bool RealVector2D::operator==(RealVector2D const& other) const
{
    return x == other.x && y == other.y;
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
