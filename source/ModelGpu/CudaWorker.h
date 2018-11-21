#pragma once

#include <mutex>
#include <QObject>

#include "ModelBasic/ChangeDescriptions.h"
#include "CudaInterface.cuh"
#include "DefinitionsImpl.h"

class CudaWorker
	: public QObject
{
	Q_OBJECT
public:
	CudaWorker(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~CudaWorker();

	void init(SpaceProperties* metric);

	void requireData();
	Q_SIGNAL void dataObtained();	//only for queued connection (due to mutex)
	void lockData();
	void unlockData();
	SimulationDataForAccess& retrieveData();
	void updateData();

	bool isSimulationRunning();
	void setFlagStopAfterNextTimestep(bool value);

	Q_SLOT void runSimulation();		
	Q_SIGNAL void timestepCalculated();

private:
	bool isDataRequired();
	void requireDataFinished();
	bool isDataUpdated();
	void updateDataFinished();

	bool stopAfterNextTimestep();
	void setSimulationRunning(bool running);

private:
	SpaceProperties* _spaceProp;
	bool _stopAfterNextTimestep = true;
	bool _simRunning = false;
	bool _requireData = false;
	bool _updateData = false;

	std::mutex _mutexForFlags;
	std::mutex _mutexForData;
	SimulationDataForAccess _cudaData;
	SimulationDataManager* _simDataManager = nullptr;
};