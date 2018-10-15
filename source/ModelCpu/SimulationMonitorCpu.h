#pragma once
#include <QObject>

#include "ModelBasic/SimulationMonitor.h"

#include "Definitions.h"

class SimulationMonitorCpu
	: public SimulationMonitor
{
	Q_OBJECT

public:
	SimulationMonitorCpu(QObject * parent = nullptr) : SimulationMonitor(parent) {}
	virtual ~SimulationMonitorCpu() = default;

	virtual void init(SimulationControllerCpu* controller) = 0;

};
