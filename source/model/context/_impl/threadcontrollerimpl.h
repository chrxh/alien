#ifndef THREADCONTROLLERIMPL_H
#define THREADCONTROLLERIMPL_H

#include "model/context/threadcontroller.h"

class ThreadControllerImpl
	: public ThreadController
{
	Q_OBJECT
public:
	ThreadControllerImpl(QObject* parent = nullptr);
	virtual ~ThreadControllerImpl();

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

#endif // THREADCONTROLLERIMPL_H
