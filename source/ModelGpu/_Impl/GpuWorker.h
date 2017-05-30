#pragma once

#include <QObject>

class GpuWorker
	: public QObject
{
	Q_OBJECT
public:
	GpuWorker(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~GpuWorker();

	virtual void init();

	Q_SLOT void calculateTimestep();
	Q_SIGNAL void timestepCalculated();
};