#include "Definitions.h"
#include "model/entities/CellCluster.h"
#include "model/entities/Cell.h"

std::size_t CellClusterHash::operator()(CellCluster* const& s) const
{
	return s->getId();
}

std::size_t CellHash::operator()(Cell* const& s) const
{
    return s->getId();
}

bool operator==(IntVector2D const& vec1,  IntVector2D const& vec2)
{
	return vec1.x == vec2.x && vec1.y == vec2.y;
}

std::ostream& operator << (std::ostream& os, const IntVector2D& vec)
{
	os << "(" << vec.x << ", " << vec.y << ")";
	return os;
}
