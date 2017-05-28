#pragma once

#include "Model/SpaceMetricApi.h"

class SpaceMetric
	: public SpaceMetricApi
{
	Q_OBJECT
public:
	SpaceMetric(QObject* parent = nullptr) : SpaceMetricApi(parent) {}
	virtual ~SpaceMetric() = default;

	virtual void init(IntVector2D size) = 0;
	virtual SpaceMetric* clone(QObject* parent = nullptr) const = 0;

	virtual void correctPosition(QVector2D& pos) const = 0;
	virtual IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const = 0;
	virtual void correctDisplacement(QVector2D& displacement) const = 0;
	virtual QVector2D displacement(QVector2D fromPoint, QVector2D toPoint) const = 0;
	virtual qreal distance(QVector2D fromPoint, QVector2D toPoint) const = 0;
	virtual QVector2D correctionIncrement (QVector2D pos1, QVector2D pos2) const = 0;

	virtual void serializePrimitives(QDataStream& stream) const = 0;
	virtual void deserializePrimitives(QDataStream& stream) = 0;
};

