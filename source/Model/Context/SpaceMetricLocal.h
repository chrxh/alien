#pragma once

#include "Model/SpaceMetric.h"

class SpaceMetricLocal
	: public SpaceMetric
{
	Q_OBJECT
public:
	SpaceMetricLocal(QObject* parent = nullptr) : SpaceMetric(parent) {}
	virtual ~SpaceMetricLocal() = default;

	virtual void init(IntVector2D size) = 0;
	virtual SpaceMetricLocal* clone(QObject* parent = nullptr) const = 0;

	virtual IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const = 0;
	virtual void correctDisplacement(QVector2D& displacement) const = 0;
	virtual QVector2D displacement(QVector2D fromPoint, QVector2D toPoint) const = 0;
	virtual qreal distance(QVector2D fromPoint, QVector2D toPoint) const = 0;
	virtual QVector2D correctionIncrement (QVector2D pos1, QVector2D pos2) const = 0;

	virtual void truncatePosition(IntVector2D& pos) const = 0;

	virtual void serializePrimitives(QDataStream& stream) const = 0;
	virtual void deserializePrimitives(QDataStream& stream) = 0;
};

