#ifndef UNITTHREADCONTROLLER_H
#define UNITTHREADCONTROLLER_H

#include "model/Definitions.h"

class UnitThreadController
	: public QObject
{
	Q_OBJECT
public:
	UnitThreadController(QObject* parent) : QObject(parent) {}
	virtual ~UnitThreadController() {}

	virtual void init(int maxRunningThreads) = 0;

	virtual void registerUnit(Unit* unit) = 0;
	virtual void start() = 0;

public slots:
	virtual void calcNextTimestep() = 0;

signals:
	void timestepFinished();
};

#endif // UNITTHREADCONTROLLER_H
