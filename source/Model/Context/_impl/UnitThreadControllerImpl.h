#ifndef UNITTHREADCONTROLLERIMPL_H
#define UNITTHREADCONTROLLERIMPL_H

#include <QSignalMapper>
#include <QMutex>
#include "gtest/gtest_prod.h"

#include "model/Context/UnitThreadController.h"

#include "DefinitionsImpl.h"
#include "SignalWrapper.h"

class MODEL_EXPORT UnitThreadControllerImpl
	: public UnitThreadController
{
	Q_OBJECT
public:
	UnitThreadControllerImpl(QObject* parent = nullptr);
	virtual ~UnitThreadControllerImpl();

	virtual void init(int maxRunningThreads) override;

	virtual void registerUnit(Unit* unit) override;
	virtual void start() override;

	virtual void registerObserver(UnitObserver* observer) override;
	virtual void unregisterObserver(UnitObserver* observer) override;

	Q_SLOT virtual bool calculateTimestep() override;

	virtual bool isNoThreadWorking() const override;

private:
	Q_SLOT void threadFinishedCalculation(QObject* sender);

	void updateDependencies();
	void terminateThreads();
	void startThreads();
	bool areAllThreadsFinished() const;
	void setAllUnitsReady();
	void searchAndExecuteReadyThreads();
	void notifyObservers() const;

	int _maxRunningThreads = 1;
	int _runningThreads = 0;
	struct UnitThreadSignal {
		Unit* unit;
		UnitThread* thr;
		SignalWrapper* calcSignal;
	};
	vector<UnitThreadSignal> _threadsAndCalcSignals;
	map<UnitContext*, UnitThread*> _threadsByContexts;
	QSignalMapper* _signalMapper = nullptr;

	vector<UnitObserver*> _observers;

	FRIEND_TEST(UnitThreadControllerImplTest, testStates);
	FRIEND_TEST(UnitThreadControllerImplTest, testStatesWithFinished);
	FRIEND_TEST(MultithreadingTest, testThreads);
};

#endif // UNITTHREADCONTROLLERIMPL_H
