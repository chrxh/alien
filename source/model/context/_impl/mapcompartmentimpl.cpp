#include "MapCompartmentImpl.h"
#include "model/context/UnitContext.h"

MapCompartmentImpl::MapCompartmentImpl(QObject * parent)
	: MapCompartment(parent)
{
}

void MapCompartmentImpl::init(IntRect mapRect)
{
	_rect = mapRect;
	_size = { _rect.p2.x - _rect.p1.x + 1, _rect.p2.y - _rect.p1.y + 1};
}

IntVector2D MapCompartmentImpl::getSize() const
{
	return _size;
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

bool MapCompartmentImpl::isPointInCompartment(IntVector2D const & intPos) const
{
	return intPos.x >= _rect.p1.x && intPos.y >= _rect.p1.y && intPos.x <= _rect.p2.x && intPos.y <= _rect.p2.y;
}

UnitContext * MapCompartmentImpl::getNeighborContext(IntVector2D const & intPos) const
{
	for (auto const& contextByLocation : _contextsByLocations) {
		auto context = contextByLocation.second;
		if (context->getMapCompartment()->isPointInCompartment(intPos)) {
			return context;
		}
	}
	return nullptr;
}

IntVector2D MapCompartmentImpl::convertAbsToRelPosition(IntVector2D const& intPos) const
{
	return{ intPos.x - intPos.x / _size.x * _size.x, intPos.y - intPos.y / _size.y * _size.y };
}

