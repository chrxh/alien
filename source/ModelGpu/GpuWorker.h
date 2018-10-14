#pragma once

#include <mutex>
#include <QObject>

#include "ModelBasic/ChangeDescriptions.h"
#include "CudaInterface.cuh"
#include "DefinitionsImpl.h"

class GpuWorker
	: public QObject
{
	Q_OBJECT
public:
	GpuWorker(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~GpuWorker();

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