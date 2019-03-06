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

	void requireData(IntRect const& rect);
	Q_SIGNAL void dataObtained();	//only for queued connection (due to mutex)
	void lockData();
	void unlockData();
	SimulationAccessTO* retrieveData();
	void updateData();

	bool isSimulationRunning();
	void stopAfterNextTimestep(bool value);
	void restrictTimestepsPerSecond(optional<int> tps);

	Q_SLOT void runSimulation();		
	Q_SIGNAL void timestepCalculated();

private:
	bool isDataRequired();
	void requireDataFinished();
	bool isDataUpdated();
	void updateDataFinished();
	optional<int> getTps();

	bool stopAfterNextTimestep();
	void setSimulationRunning(bool running);

private:
	SpaceProperties* _spaceProp;
	bool _stopAfterNextTimestep = true;
	optional<int> _tps;
	bool _simRunning = false;
	bool _requireData = false;
	bool _updateData = false;
	IntRect _requiredRect;

	std::mutex _mutexForFlags;
	std::mutex _mutexForData;
	SimulationAccessTO* _cudaData;
	CudaSimulation* _cudaSimulation = nullptr;
};