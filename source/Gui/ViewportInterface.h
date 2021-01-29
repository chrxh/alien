#pragma once

#include <QRectF>

#include "Gui/Definitions.h"

class ViewportInterface
	: public QObject
{
	Q_OBJECT
public:
	ViewportInterface(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~ViewportInterface() = default;

	virtual QRectF getRect() const = 0;
};
