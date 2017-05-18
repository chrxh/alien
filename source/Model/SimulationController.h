#ifndef SIMULATIONCONTROLLER_H
#define SIMULATIONCONTROLLER_H

#include "Definitions.h"

class SimulationController
	: public QObject
{
    Q_OBJECT
public:
	SimulationController(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationController() = default;

    virtual void init(SimulationContextApi* context) = 0;
    virtual void setRun(bool run) = 0;
	virtual void calculateSingleTimestep() = 0;
	virtual SimulationContextApi* getContext() const = 0;

	Q_SIGNAL void nextFrameCalculated();
	Q_SIGNAL void nextTimestepCalculated();
	Q_SIGNAL void updateTimestepsPerSecond(int value);
};

#endif // SIMULATIONCONTROLLER_H
