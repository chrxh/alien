#pragma once

#include <mutex>
#include <QObject>

#include "Model/Api/ChangeDescriptions.h"
#include "Cuda/CudaInterface.cuh"
#include "DefinitionsImpl.h"

class WorkerForGpu
	: public QObject
{
	Q_OBJECT
public:
	WorkerForGpu(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~WorkerForGpu();

	virtual void init(SpaceProperties* metric);
	virtual void requireData();
	Q_SIGNAL void dataReadyToRetrieve();
	virtual DataForAccess retrieveData();
	virtual void ptrCorrectionForRetrievedData();
	virtual void lockData();
	virtual void unlockData();

	virtual bool isSimulationRunning();
	virtual void setFlagStopAfterNextTimestep(bool value);

	Q_SLOT void runSimulation();
	Q_SIGNAL void timestepCalculated();

private:
	SpaceProperties* _metric;

	bool _simRunning = false;
	bool _stopAfterNextTimestep = true;
	bool _requireData = false;
	std::mutex _mutex;
	DataForAccess _cudaData;
};