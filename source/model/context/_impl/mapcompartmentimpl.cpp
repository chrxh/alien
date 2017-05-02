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

std::vector<UnitContext*> MapCompartmentImpl::getNeighborContexts() const
{
	std::vector<UnitContext*> result;
	for (auto const& contextByLocation : _contextsByLocations) {
		result.push_back(contextByLocation.second);
	}
	return result;
}

UnitContext* MapCompartmentImpl::getNeighborContext(IntVector2D pos) const
{
	return nullptr;
}
