#include "definitions.h"
#include "model/entities/cellcluster.h"
#include "model/entities/cell.h"

std::size_t CellClusterHash::operator()(CellCluster* const& s) const
{
	return s->getId();
}

std::size_t CellHash::operator()(Cell* const& s) const
{
    return s->getId();
}
