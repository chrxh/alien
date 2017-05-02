#ifndef UNITTHREADCONTROLLERIMPL_H
#define UNITTHREADCONTROLLERIMPL_H

#include "DefinitionsImpl.h"
#include "model/context/UnitThreadController.h"
#include "SignalWrapper.h"

class UnitThreadControllerImpl
	: public UnitThreadController
{
	Q_OBJECT
public:
	UnitThreadControllerImpl(QObject* parent = nullptr);
	virtual ~UnitThreadControllerImpl();

	virtual void init(int maxRunningThreads) override;

	virtual void registerUnit(Unit* unit) override;
	virtual void start() override;

private:
	void updateDependencies();
	void terminateThreads();
	void startThreads();
	void searchAndExecuteReadyThreads();

	int _maxRunningThreads = 1;

	struct UnitThreadSignal {
		UnitThread* thr;
		SignalWrapper* signal;
	};
	std::vector<UnitThreadSignal> _threadsAndCalcSignals;
	std::map<UnitContext*, UnitThread*> _threadsByContexts;

};

#endif // UNITTHREADCONTROLLERIMPL_H
