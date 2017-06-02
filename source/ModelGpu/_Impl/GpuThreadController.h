#pragma once

#include <QThread>

#include "Model/Definitions.h"
#include "DefinitionsImpl.h"

class GpuThreadController
	: public QObject
{
	Q_OBJECT
public:
	GpuThreadController(QObject* parent = nullptr);
	virtual ~GpuThreadController();

	void init(SpaceMetricApi *metric);

	virtual void registerObserver(GpuObserver* observer);
	virtual void unregisterObserver(GpuObserver* observer);
	virtual void notifyObserver();
	virtual GpuWorker* getGpuWorker() const;
	virtual bool isGpuThreadWorking() const;

	void calculateTimestep();
	Q_SIGNAL void timestepCalculated();

private:
	Q_SIGNAL void calculateTimestepWithGpu();
	Q_SLOT void timestepCalculatedWithGpu();

	SpaceMetricApi *_metric = nullptr;

	QThread _thread;
	GpuWorker* _worker = nullptr;
	bool _gpuThreadWorking = false;

	vector<GpuObserver*> _observers;
};
