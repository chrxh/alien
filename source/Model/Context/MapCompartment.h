#ifndef MAPCOMPARTMENT_H
#define MAPCOMPARTMENT_H

#include "model/Definitions.h"

class MapCompartment
	: public QObject
{
	Q_OBJECT
public:
	MapCompartment(QObject* parent) : QObject(parent) {}
	virtual ~MapCompartment() = default;

	virtual void init(IntRect mapRect) = 0;

	virtual IntVector2D getSize() const = 0;
	enum class RelativeLocation {
		UpperLeft, Upper, UpperRight, 
		Left, Right, 
		LowerLeft, Lower, LowerRight,
	};
	virtual void registerNeighborContext(RelativeLocation location, UnitContext* context) = 0;
	virtual vector<UnitContext*> getNeighborContexts() const = 0;
	virtual bool isPointInCompartment(IntVector2D const& intPos) const = 0;
	virtual UnitContext* getNeighborContext(IntVector2D const& intPos) const = 0;

	inline void convertAbsToRelPosition(IntVector2D & intPos) const;

protected:
	IntVector2D _size;
};

/****************** inline methods ********************/
void MapCompartment::convertAbsToRelPosition(IntVector2D & intPos) const
{
	intPos.x = intPos.x - intPos.x / _size.x * _size.x;
	intPos.y = intPos.y - intPos.y / _size.y * _size.y;
}


#endif // MAPCOMPARTMENT_H
