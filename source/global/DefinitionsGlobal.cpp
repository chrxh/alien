#include "Definitions.h"

bool operator==(IntVector2D const& vec1, IntVector2D const& vec2)
{
	return vec1.x == vec2.x && vec1.y == vec2.y;
}

std::ostream& operator << (std::ostream& os, const IntVector2D& vec)
{
	os << "(" << vec.x << ", " << vec.y << ")";
	return os;
}
