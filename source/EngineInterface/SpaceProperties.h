#pragma once

#include "Definitions.h"

class ENGINEINTERFACE_EXPORT SpaceProperties
	: public QObject
{
	Q_OBJECT
public:
	SpaceProperties(QObject* parent = nullptr);
	virtual ~SpaceProperties() {}

	virtual void init(IntVector2D size);
	virtual SpaceProperties* clone(QObject* parent = nullptr) const;

	virtual IntVector2D getSize() const;
	virtual IntVector2D convertToIntVector(QVector2D const &pos) const;
	virtual IntVector2D correctPositionAndConvertToIntVector(QVector2D const& pos) const;
	virtual void correctPosition(QVector2D& pos) const;
	virtual void correctPosition(IntVector2D & pos) const;
	virtual void correctDisplacement(QVector2D& displacement) const;

	virtual QVector2D correctionIncrement(QVector2D pos1, QVector2D pos2) const;
	
	virtual void truncatePosition(IntVector2D& pos) const;
	virtual void truncateRect(IntRect& rect) const;
	virtual QVector2D displacement(QVector2D fromPoint, QVector2D toPoint) const;
	virtual qreal distance(QVector2D fromPoint, QVector2D toPoint) const;
	virtual IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const;

private:
	inline void correctPositionInline(IntVector2D & pos) const;

	IntVector2D _size{ 0, 0 };
};

void SpaceProperties::correctPositionInline(IntVector2D & pos) const
{
	pos.x = ((pos.x % _size.x) + _size.x) % _size.x;
	pos.y = ((pos.y % _size.y) + _size.y) % _size.y;
}

