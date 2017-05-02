#ifndef THREADCONTROLLER_H
#define THREADCONTROLLER_H

#include "model/definitions.h"

class ThreadController
	: public QObject
{
	Q_OBJECT
public:
	ThreadController(QObject* parent) : QObject(parent) {}
	virtual ~ThreadController() {}

	virtual void init(int maxRunningThreads) = 0;

	virtual void registerUnit(Unit* unit) = 0;
	virtual void start() = 0;
};

#endif // THREADCONTROLLER_H
