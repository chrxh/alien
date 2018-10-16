#pragma once

#include <mutex>
#include <QObject>

#include "ModelBasic/ChangeDescriptions.h"
#include "CudaInterface.cuh"
#include "DefinitionsImpl.h"

class CudaBridge
	: public QObject
{
	Q_OBJECT
public:
	CudaBridge(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~CudaBridge();

	virtual void init(SpaceProperties* metric);
	virtual void requireData();
	virtual bool isDataRequired();
	Q_SIGNAL void dataObtained();

	virtual SimulationDataForAccess retrieveData();
	virtual void lockData();
	virtual void unlockData();

	virtual bool isSimulationRunning();
	virtual void setFlagStopAfterNextTimestep(bool value);

	Q_SLOT void runSimulation();
	Q_SIGNAL void timestepCalculated();

private:
	virtual void dataObtainedIntern();

private:
	SpaceProperties* _spaceProp;

	bool _simRunning = false;
	bool _stopAfterNextTimestep = true;
	bool _requireData = false;
	std::mutex _mutexForData;
	std::mutex _mutexForRequirement;
	SimulationDataForAccess _cudaData;
};