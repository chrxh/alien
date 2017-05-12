#include "Definitions.h"
#include "model/Entities/CellCluster.h"
#include "model/Entities/Cell.h"

std::size_t CellClusterHash::operator()(CellCluster* const& s) const
{
	return s->getId();
}

std::size_t CellHash::operator()(Cell* const& s) const
{
    return s->getId();
}

