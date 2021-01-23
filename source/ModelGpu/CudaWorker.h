#pragma once

#include <mutex>
#include <QThread>

#include "ModelBasic/ChangeDescriptions.h"
#include "AccessTOs.cuh"
#include "DefinitionsImpl.h"

class CudaWorker
	: public QObject
{
	Q_OBJECT
public:

	CudaWorker(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~CudaWorker();

    void init(
        SpaceProperties* space,
        int timestep,
        SimulationParameters const& parameters,
        CudaConstants const& cudaConstants,
        NumberGenerator* numberGenerator);
    void terminateWorker();
	bool isSimulationRunning();
    int getTimestep() const;
    void setTimestep(int timestep);

	void addJob(CudaJob const& job);
	vector<CudaJob> getFinishedJobs(string const& originId);
	Q_SIGNAL void jobsFinished();

	Q_SIGNAL void timestepCalculated();

	Q_SLOT void run();

private:
	void processJobs();
	bool isTerminate();

private:
	SpaceProperties* _space = nullptr;
	CudaSimulation* _cudaSimulation = nullptr;
    NumberGenerator* _numberGenerator = nullptr;

	mutable std::mutex _mutex;
	std::condition_variable _condition;
	list<CudaJob> _jobs;
	vector<CudaJob> _finishedJobs;

	bool _simulationRunning = false;
	bool _terminate = false;
	optional<int> _tpsRestriction;
};