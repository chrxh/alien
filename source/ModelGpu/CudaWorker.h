#pragma once

#include <mutex>
#include <QThread>

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

	void init(SpaceProperties* space);
	void terminateWorker();
	bool isSimulationRunning();

	void addJob(CudaJob const& job);
	vector<CudaJob> getFinishedJobs(string const& originId);
	Q_SIGNAL void jobsFinished();

	Q_SIGNAL void timestepCalculated();

	Q_SLOT void run();

/*
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
*/

private:
	void processJobs();
	bool isTerminate();

private:
	SpaceProperties* _space = nullptr;
	CudaSimulation* _cudaSimulation = nullptr;

	std::mutex _mutex;
//	std::mutex _conditionMutex;
	std::condition_variable _condition;
	vector<CudaJob> _jobs;
	vector<CudaJob> _finishedJobs;

	bool _simulationRunning = false;
	bool _terminate = false;

	/*
	bool _stopAfterNextTimestep = true;
	optional<int> _tps;
	bool _simRunning = false;
	bool _requireData = false;
	bool _updateData = false;
	IntRect _requiredRect;

	std::mutex _mutexForFlags;
	std::mutex _mutexForData;
	SimulationAccessTO* _cudaData;
*/
};