#include "MapCompartmentImpl.h"

MapCompartmentImpl::MapCompartmentImpl(QObject * parent)
	: MapCompartment(parent)
{
}

void MapCompartmentImpl::init(SpaceMetric * metric, IntRect mapRect)
{
}

void MapCompartmentImpl::registerNeighborContext(RelativeLocation location, UnitContext * context)
{
}

UnitContext * MapCompartmentImpl::getNeighborContext(RelativeLocation location) const
{
	return nullptr;
}

UnitContext * MapCompartmentImpl::getNeighborContext(IntVector2D pos) const
{
	return nullptr;
}
