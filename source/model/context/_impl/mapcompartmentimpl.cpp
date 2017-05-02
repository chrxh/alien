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
	_contextsByLocations[location] = context;
}

UnitContext* MapCompartmentImpl::getNeighborContext(RelativeLocation location) const
{
	auto contextByLocation = _contextsByLocations.find(location);
	if (contextByLocation != _contextsByLocations.end()) {
		return contextByLocation->second;
	}
	return nullptr;
}

UnitContext* MapCompartmentImpl::getNeighborContext(IntVector2D pos) const
{
	return nullptr;
}
