#ifndef SPACEMETRIC_H
#define SPACEMETRIC_H

#include "model/Definitions.h"

class SpaceMetric
	: public QObject
{
	Q_OBJECT
public:
	SpaceMetric(QObject* parent) : QObject(parent) {}
	virtual ~SpaceMetric() = default;

	virtual void init(IntVector2D size) = 0;
	virtual SpaceMetric* clone(QObject* parent = nullptr) const = 0;

	virtual IntVector2D getSize() const = 0;

	virtual void correctPosition(QVector2D& pos) const = 0;
	virtual IntVector2D correctPositionWithIntPrecision(QVector2D const& pos) const = 0;
	virtual IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const = 0;
	virtual void correctDisplacement(QVector2D& displacement) const = 0;
	virtual QVector2D displacement(QVector2D fromPoint, QVector2D toPoint) const = 0;
	virtual qreal distance(QVector2D fromPoint, QVector2D toPoint) const = 0;
	virtual QVector2D correctionIncrement (QVector2D pos1, QVector2D pos2) const = 0;

	virtual void serializePrimitives(QDataStream& stream) const = 0;
	virtual void deserializePrimitives(QDataStream& stream) = 0;
};

#endif // SPACEMETRIC_H
