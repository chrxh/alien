#pragma once

#include "gui/Definitions.h"

class ViewportInterface
	: public QObject
{
	Q_OBJECT
public:
	ViewportInterface(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~ViewportInterface() = default;

	virtual void setModeToUpdate() = 0;
	virtual void setModeToNoUpdate() = 0;

	virtual QRectF getRect() const = 0;
	virtual QVector2D getCenter() const = 0;
};
