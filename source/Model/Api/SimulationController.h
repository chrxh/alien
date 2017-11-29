#pragma once

#include "Definitions.h"

class MODEL_EXPORT SimulationController
	: public QObject
{
    Q_OBJECT
public:
	SimulationController(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationController() = default;

    virtual void setRun(bool run) = 0;
	virtual void calculateSingleTimestep() = 0;
	virtual SimulationContext* getContext() const = 0;

	Q_SIGNAL void nextFrameCalculated();
	Q_SIGNAL void nextTimestepCalculated();
	Q_SIGNAL void updateTimestepsPerSecond(int value);
};

