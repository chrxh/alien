#pragma once

#include "Model/Api/SpaceProperties.h"

class SpacePropertiesLocal
	: public SpaceProperties
{
	Q_OBJECT
public:
	SpacePropertiesLocal(QObject* parent = nullptr) : SpaceProperties(parent) {}
	virtual ~SpacePropertiesLocal() = default;

	virtual void init(IntVector2D size) = 0;
	virtual SpacePropertiesLocal* clone(QObject* parent = nullptr) const = 0;

	virtual IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const = 0;
	virtual void correctDisplacement(QVector2D& displacement) const = 0;
	virtual QVector2D displacement(QVector2D fromPoint, QVector2D toPoint) const = 0;
	virtual qreal distance(QVector2D fromPoint, QVector2D toPoint) const = 0;
	virtual QVector2D correctionIncrement (QVector2D pos1, QVector2D pos2) const = 0;

	virtual void truncatePosition(IntVector2D& pos) const = 0;
};

