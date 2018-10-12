#pragma once

#include "ModelBasic/SpaceProperties.h"

class SpacePropertiesImpl
	: public SpaceProperties
{
public:
	SpacePropertiesImpl(QObject* parent = nullptr);
	virtual ~SpacePropertiesImpl() {}

	virtual IntVector2D getSize() const override;

	virtual void correctPosition(QVector2D& pos) const override;
	virtual void correctPosition(IntVector2D & pos) const override;
	virtual IntVector2D convertToIntVector(QVector2D const &pos) const override;
	virtual IntVector2D correctPositionAndConvertToIntVector(QVector2D const& pos) const override;
	virtual void correctDisplacement(QVector2D& displacement) const override;
	virtual QVector2D correctionIncrement(QVector2D pos1, QVector2D pos2) const;

	virtual void init(IntVector2D size);
	virtual SpacePropertiesImpl* clone(QObject* parent = nullptr) const;

	virtual void truncatePosition(IntVector2D& pos) const;
	virtual QVector2D displacement(QVector2D fromPoint, QVector2D toPoint) const;
	virtual qreal distance(QVector2D fromPoint, QVector2D toPoint) const;
	virtual IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const;

private:
	inline void correctPositionInline(IntVector2D & pos) const;

	IntVector2D _size{ 0, 0 };
};

void SpacePropertiesImpl::correctPositionInline(IntVector2D & pos) const
{
	pos.x = ((pos.x % _size.x) + _size.x) % _size.x;
	pos.y = ((pos.y % _size.y) + _size.y) % _size.y;
}
