#ifndef TORUSTOPOLOGY_H
#define TORUSTOPOLOGY_H

#include "model/context/spacemetric.h"

class SpaceMetricImpl
	: public SpaceMetric
{
public:
	SpaceMetricImpl(QObject* parent = nullptr);
	virtual ~SpaceMetricImpl() {}

	virtual void init(IntVector2D size);
	virtual SpaceMetric* clone(QObject* parent = nullptr) const override;

	virtual IntVector2D getSize() const override;

	virtual void correctPosition(QVector3D& pos) const override;
	virtual IntVector2D correctPositionWithIntPrecision(QVector3D const& pos) const override;
	virtual IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const override;
	virtual void correctDisplacement(QVector3D& displacement) const override;
	virtual QVector3D displacement(QVector3D fromPoint, QVector3D toPoint) const override;
	virtual qreal distance(QVector3D fromPoint, QVector3D toPoint) const override;
	virtual QVector3D correctionIncrement(QVector3D pos1, QVector3D pos2) const override;

	virtual void serializePrimitives(QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream) override;

private:
	inline void correctPosition(IntVector2D & pos) const;

	IntVector2D _size{ 0, 0 };
};

void SpaceMetricImpl::correctPosition(IntVector2D & pos) const
{
	pos.x = ((pos.x % _size.x) + _size.x) % _size.x;
	pos.y = ((pos.y % _size.y) + _size.y) % _size.y;
}

#endif // TORUSTOPOLOGY_H
