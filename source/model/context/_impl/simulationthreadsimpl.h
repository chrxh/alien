#ifndef SIMULATIONTHREADSIMPL_H
#define SIMULATIONTHREADSIMPL_H

#include "model/context/simulationthreads.h"

class SimulationThreadsImpl
	: public SimulationThreads
{
	Q_OBJECT
public:
	SimulationThreadsImpl(QObject* parent = nullptr);
	virtual ~SimulationThreadsImpl();

	virtual void init(int maxRunningThreads) override;

	virtual void registerUnit(SimulationUnit* unit) override;
	virtual void start() override;

private:
	void updateDependencies();
	void terminateThreads();

	int _maxRunningThreads = 1;
	std::vector<QThread*> _threads;
	std::map<QThread*, std::vector<QThread*>> _dependencies;
	std::map<SimulationUnitContext*, QThread*> _threadsByContexts;
};

#endif // SIMULATIONTHREADSIMPL_H
