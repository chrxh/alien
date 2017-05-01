#include "mapcompartmentimpl.h"

MapCompartmentImpl::MapCompartmentImpl(QObject * parent)
	: MapCompartment(parent)
{
}

void MapCompartmentImpl::init(SpaceMetric * metric, IntRect mapRect)
{
}

void MapCompartmentImpl::registerNeighborContext(RelativeLocation location, SimulationUnitContext * context)
{
}

SimulationUnitContext * MapCompartmentImpl::getNeighborContext(RelativeLocation location) const
{
	return nullptr;
}

SimulationUnitContext * MapCompartmentImpl::getNeighborContext(IntVector2D pos) const
{
	return nullptr;
}
