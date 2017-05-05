#ifndef UNITTHREADCONTROLLER_H
#define UNITTHREADCONTROLLER_H

#include "model/Definitions.h"

class UnitThreadController
	: public QObject
{
	Q_OBJECT
public:
	UnitThreadController(QObject* parent) : QObject(parent) {}
	virtual ~UnitThreadController() = default;

	virtual void init(int maxRunningThreads) = 0;

	virtual void lock() = 0;
	virtual void unlock() = 0;

	virtual void registerUnit(Unit* unit) = 0;
	virtual void start() = 0;

	Q_SLOT virtual bool calculateTimestep() = 0;
	Q_SIGNAL void timestepCalculated();
};

#endif // UNITTHREADCONTROLLER_H
