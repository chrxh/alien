#pragma once

#include <QObject>

#include "Model/Entities/Descriptions.h"
#include "Cuda/CudaShared.cuh"
#include "DefinitionsImpl.h"

class GpuWorker
	: public QObject
{
	Q_OBJECT
public:
	GpuWorker(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~GpuWorker();

	virtual void init(SpaceMetricApi* metric);
	virtual void requireData();
	Q_SIGNAL void dataReadyToRetrieve();
	virtual CudaData retrieveData();

	virtual bool isSimulationRunning();
	virtual RunningMode getMode();
	virtual void setMode(RunningMode mode);

	Q_SLOT void runSimulation();
	Q_SIGNAL void timestepCalculated();

private:
	SpaceMetricApi* _metric;

	bool _simRunning = false;
	RunningMode _mode = RunningMode::StopAfterNextTimestep;
};