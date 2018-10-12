#pragma once

#include "ModelBasic/Definitions.h"
#include "UnitContext.h"

class MapCompartment
	: public QObject
{
	Q_OBJECT
public:
	MapCompartment(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~MapCompartment() = default;

	inline void init(IntRect mapRect);

	inline IntVector2D getSize() const;
	enum class RelativeLocation {
		UpperLeft, Upper, UpperRight, 
		Left, Right, 
		LowerLeft, Lower, LowerRight,
	};
	inline void registerNeighborContext(RelativeLocation location, UnitContext* context);
	inline vector<UnitContext*> getNeighborContexts() const;
	inline bool isPointInCompartment(IntVector2D const& intPos) const;
	inline UnitContext* getNeighborContext(IntVector2D const& intPos) const;

	inline void convertAbsToRelPosition(IntVector2D & intPos) const;

protected:
	IntRect _rect;
	IntVector2D _size;
	map<RelativeLocation, UnitContext*> _contextsByLocations;
};

/****************** inline methods ********************/
void MapCompartment::init(IntRect mapRect)
{
	_rect = mapRect;
	_size = { _rect.p2.x - _rect.p1.x + 1, _rect.p2.y - _rect.p1.y + 1 };
}

IntVector2D MapCompartment::getSize() const
{
	return _size;
}

void MapCompartment::registerNeighborContext(RelativeLocation location, UnitContext * context)
{
	_contextsByLocations[location] = context;
}

vector<UnitContext*> MapCompartment::getNeighborContexts() const
{
	vector<UnitContext*> result;
	for (auto const& contextByLocation : _contextsByLocations) {
		result.push_back(contextByLocation.second);
	}
	return result;
}

bool MapCompartment::isPointInCompartment(IntVector2D const & intPos) const
{
	return intPos.x >= _rect.p1.x && intPos.y >= _rect.p1.y && intPos.x <= _rect.p2.x && intPos.y <= _rect.p2.y;
}

UnitContext * MapCompartment::getNeighborContext(IntVector2D const & intPos) const
{
	for (auto const& contextByLocation : _contextsByLocations) {
		auto context = contextByLocation.second;
		if (context->getMapCompartment()->isPointInCompartment(intPos)) {
			return context;
		}
	}
	return nullptr;
}

void MapCompartment::convertAbsToRelPosition(IntVector2D & intPos) const
{
	intPos.x = intPos.x - intPos.x / _size.x * _size.x;
	intPos.y = intPos.y - intPos.y / _size.y * _size.y;
}
