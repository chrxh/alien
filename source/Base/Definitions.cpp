#include "Definitions.h"

#include <ostream>

IntVector2D::IntVector2D(std::initializer_list<int> l)
{
    auto it = l.begin();
    x = *it++;
    y = *it;
}

bool IntVector2D::operator==(IntVector2D const& vec) const
{
    return x == vec.x && y == vec.y;
}

void IntVector2D::operator-=(IntVector2D const& vec)
{
    x -= vec.x;
    y -= vec.y;
}

std::ostream& operator<<(std::ostream& os, const IntVector2D& vec)
{
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}
