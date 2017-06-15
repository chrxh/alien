#pragma once

#include <QObject>

#include "Model/Entities/Descriptions.h"

class GpuWorker
	: public QObject
{
	Q_OBJECT
public:
	GpuWorker(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~GpuWorker();

	virtual void init(SpaceMetricApi* metric);
	virtual void getData(IntRect const &rect, ResolveDescription const &resolveDesc, DataDescription &result);
	virtual void getImage(IntRect const &rect, QImage* target);

	Q_SLOT void calculateTimestep();
	Q_SIGNAL void timestepCalculated();

private:
	SpaceMetricApi* _metric;
};