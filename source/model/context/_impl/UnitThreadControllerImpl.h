#ifndef UNITTHREADCONTROLLERIMPL_H
#define UNITTHREADCONTROLLERIMPL_H

#include <QSignalMapper>
#include "gtest/gtest_prod.h"

#include "model/context/UnitThreadController.h"

#include "DefinitionsImpl.h"
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

	Q_SLOT virtual void calculateTimestep() override;

private:
	Q_SLOT void threadFinishedCalculation(QObject* sender);

	void updateDependencies();
	void terminateThreads();
	void startThreads();
	bool areAllUnitsFinished();
	void setAllUnitsReady();
	void searchAndExecuteReadyThreads();

	int _maxRunningThreads = 1;

	struct UnitThreadSignal {
		UnitThread* thr;
		SignalWrapper* signal;
	};
	std::vector<UnitThreadSignal> _threadsAndCalcSignals;
	std::map<UnitContext*, UnitThread*> _threadsByContexts;
	QSignalMapper* _signalMapper = nullptr;

	FRIEND_TEST(UnitThreadControllerImplTest, testStates);
	FRIEND_TEST(UnitThreadControllerImplTest, testStatesWithFinished);
	FRIEND_TEST(MultithreadingTest, testThreads);
};

#endif // UNITTHREADCONTROLLERIMPL_H
