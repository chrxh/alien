#pragma once

#include "Model/Api/Definitions.h"

class MODEL_EXPORT UnitThreadController
	: public QObject
{
	Q_OBJECT
public:
	UnitThreadController(QObject* parent) : QObject(parent) {}
	virtual ~UnitThreadController() = default;

	virtual void init(int maxRunningThreads) = 0;

	virtual void registerUnit(Unit* unit) = 0;
	virtual void start() = 0;

	virtual void registerObserver(UnitObserver* observer) = 0;
	virtual void unregisterObserver(UnitObserver* observer) = 0;

	Q_SLOT virtual bool calculateTimestep() = 0;
	Q_SIGNAL void timestepCalculated();

	virtual bool isNoThreadWorking() const = 0;
	virtual int getMaxRunningThreads() const = 0;
};
