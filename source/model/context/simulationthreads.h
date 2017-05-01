#ifndef SIMULATIONTHREADS_H
#define SIMULATIONTHREADS_H

#include "model/definitions.h"

class SimulationThreads
	: public QObject
{
	Q_OBJECT
public:
	SimulationThreads(QObject* parent) : QObject(parent) {}
	virtual ~SimulationThreads() {}

	virtual void init(int maxRunningThreads) = 0;

	virtual void registerUnit(SimulationUnit* unit) = 0;
	virtual void start() const = 0;

private:
	int maxRunningThreads = 1;
	std::vector<QThread*> _threads;
};

#endif // SIMULATIONTHREADS_H
