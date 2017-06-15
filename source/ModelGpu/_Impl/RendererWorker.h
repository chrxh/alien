#pragma once

#include <QObject>

#include "Model/Entities/Descriptions.h"

class RendererWorker
	: public QObject
{
	Q_OBJECT
public:
	RendererWorker(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~RendererWorker();

	virtual void init(SpaceMetricApi* metric);

	Q_SLOT void calcImage(IntRect const &rect, QImage* target);
	Q_SIGNAL void imageCalculated();

private:
	SpaceMetricApi* _metric;
};