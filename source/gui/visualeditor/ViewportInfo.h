#ifndef VIEWPORTINFO_H
#define VIEWPORTINFO_H

#include "gui/Definitions.h"

class ViewportInfo
	: public QObject
{
	Q_OBJECT
public:
	ViewportInfo(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~ViewportInfo() = default;

	virtual QRectF getRect() const = 0;
	virtual QVector2D getCenter() const = 0;
};

#endif // VIEWPORTINFO_H
