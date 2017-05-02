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
