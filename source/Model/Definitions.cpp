#include "Definitions.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Cell.h"

std::size_t CellClusterHash::operator()(Cluster* const& s) const
{
	return s->getId();
}

std::size_t CellHash::operator()(Cell* const& s) const
{
    return s->getId();
}

