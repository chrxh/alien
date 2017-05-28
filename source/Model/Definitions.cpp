#include "Definitions.h"
#include "Model/Entities/CellCluster.h"
#include "Model/Entities/Cell.h"

std::size_t CellClusterHash::operator()(CellCluster* const& s) const
{
	return s->getId();
}

std::size_t CellHash::operator()(Cell* const& s) const
{
    return s->getId();
}

