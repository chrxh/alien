#include "mapcompartmentimpl.h"

MapCompartmentImpl::MapCompartmentImpl(QObject * parent)
	: MapCompartment(parent)
{
}

void MapCompartmentImpl::init(Topology * topology, IntRect mapRect)
{
}

void MapCompartmentImpl::registerNeighborContext(RelativeLocation location, SimulationUnitContext * context)
{
}

SimulationUnitContext * MapCompartmentImpl::getNeighborContext(IntVector2D pos) const
{
	return nullptr;
}
