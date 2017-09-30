#include "Definitions.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Cell.h"

std::size_t CellClusterHash::operator()(Cluster* const& s) const
{
	return s->getId();
}

std::size_t CellHash::operator()(Cell* const& s) const
{
    return s->getId();
}

