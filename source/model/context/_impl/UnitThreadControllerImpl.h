#ifndef UNITTHREADCONTROLLERIMPL_H
#define UNITTHREADCONTROLLERIMPL_H

#include "model/context/UnitThreadController.h"

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

	int _maxRunningThreads = 1;
	std::vector<QThread*> _threads;
	std::map<QThread*, std::vector<QThread*>> _dependencies;
	std::map<UnitContext*, QThread*> _threadsByContexts;
};

#endif // UNITTHREADCONTROLLERIMPL_H
