#include "Definitions.h"


std::ostream& operator << (std::ostream& os, const IntVector2D& vec)
{
	os << "(" << vec.x << ", " << vec.y << ")";
	return os;
}
