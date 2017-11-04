#pragma once

#include <QObject>

class ToolbarModel
	: public QObject
{
	Q_OBJECT
public:
	ToolbarModel(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~ToolbarModel() = default;

	QVector2D getPositionDeltaForNewEntity();

private:
	double _delta = 0.0;
};