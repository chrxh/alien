#include "MapCompartmentImpl.h"

MapCompartmentImpl::MapCompartmentImpl(QObject * parent)
	: MapCompartment(parent)
{
}

void MapCompartmentImpl::init(SpaceMetric * metric, IntRect mapRect)
{
	_rect = mapRect;
}

IntVector2D MapCompartmentImpl::getSize() const
{
	return{ _rect.p2.x - _rect.p1.x, _rect.p2.y - _rect.p1.y };
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

bool MapCompartmentImpl::isPointInCompartment(IntVector2D const &) const
{
	return false;
}

UnitContext * MapCompartmentImpl::getNeighborContext(IntVector2D const & pos) const
{
	return nullptr;
}

IntVector2D MapCompartmentImpl::convertAbsToRelPosition(IntVector2D const& pos) const
{
}

