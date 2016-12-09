#include "definitions.h"
#include "model/entities/cellcluster.h"

std::size_t CellClusterHash::operator()(CellCluster* const& s) const
{
	return s->getId();
}
