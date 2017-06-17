#pragma once

#include <mutex>
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
	virtual CudaDataForAccess retrieveData();
	virtual void unlockData();

	virtual bool isSimulationRunning();
	virtual void setFlagStopAfterNextTimestep(bool value);

	Q_SLOT void runSimulation();
	Q_SIGNAL void timestepCalculated();

private:
	SpaceMetricApi* _metric;

	bool _simRunning = false;
	bool _stopAfterNextTimestep = true;
	bool _requireData = false;
	std::mutex _mutex;
	CudaDataForAccess _cudaData;
};