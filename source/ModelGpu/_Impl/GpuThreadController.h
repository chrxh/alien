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

	virtual GpuWorker* getGpuWorker() const;

	void calculate(RunningMode mode);

	Q_SIGNAL void timestepCalculated();

private:
	Q_SIGNAL void runSimulationWithGpu();
	Q_SLOT void timestepCalculatedWithGpu();

	SpaceMetricApi *_metric = nullptr;

	QThread _thread;
	GpuWorker* _worker = nullptr;
	bool _gpuThreadWorking = false;
};
