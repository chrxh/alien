#include "Definitions.h"

#include <ostream>

#include <QtCore/QRectF>

IntVector2D::IntVector2D(std::initializer_list<int> l)
{
    auto it = l.begin();
    x = *it++;
    y = *it;
}

IntVector2D::IntVector2D(QVector2D const& vec)
    : x(static_cast<int>(vec.x()))
    , y(static_cast<int>(vec.y()))
{}

QVector2D IntVector2D::toQVector2D()
{
    return QVector2D(x, y);
}

IntVector2D& IntVector2D::restrictToRect(IntRect const& rect)
{
    if (x < rect.p1.x) {
        x = rect.p1.x;
    }
    if (y < rect.p1.y) {
        y = rect.p1.y;
    }
    if (x > rect.p2.x) {
        x = rect.p2.x;
    }
    if (y > rect.p2.y) {
        y = rect.p2.y;
    }
    return *this;
}

bool IntVector2D::operator==(IntVector2D const& vec)
{
    return x == vec.x && y == vec.y;
}

void IntVector2D::operator-=(IntVector2D const& vec)
{
    x -= vec.x;
    y -= vec.y;
}

IntRect::IntRect(std::initializer_list<IntVector2D> l)
{
    auto it = l.begin();
    p1 = *it++;
    p2 = *it;
}

IntRect::IntRect(QRectF const& rect)
    : p1({static_cast<int>(rect.left()), static_cast<int>(rect.top())})
    , p2({static_cast<int>(rect.right()), static_cast<int>(rect.bottom())})
{}


RealVector2D::RealVector2D(std::initializer_list<float> l)
{
    auto it = l.begin();
    x = *it++;
    y = *it;
}

RealVector2D::RealVector2D(QVector2D const& vec)
    : x(vec.x())
    , y(vec.y())
{}

QVector2D RealVector2D::toQVector2D()
{
    return QVector2D(x, y);
}

RealVector2D& RealVector2D::restrictToRect(RealRect const& rect)
{
    if (x < rect.p1.x) {
        x = rect.p1.x;
    }
    if (y < rect.p1.y) {
        y = rect.p1.y;
    }
    if (x > rect.p2.x) {
        x = rect.p2.x;
    }
    if (y > rect.p2.y) {
        y = rect.p2.y;
    }
    return *this;
}

bool RealVector2D::operator==(RealVector2D const& vec)
{
    return x == vec.x && y == vec.y;
}

void RealVector2D::operator-=(RealVector2D const& vec)
{
    x -= vec.x;
    y -= vec.y;
}

RealRect::RealRect(std::initializer_list<RealVector2D> l)
{
    auto it = l.begin();
    p1 = *it++;
    p2 = *it;
}

RealRect::RealRect(QRectF const& rect)
    : p1({static_cast<float>(rect.left()), static_cast<float>(rect.top())})
    , p2({static_cast<float>(rect.right()), static_cast<float>(rect.bottom())})
{}


std::ostream& operator<<(std::ostream& os, const IntVector2D& vec)
{
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}
