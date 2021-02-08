#pragma once

#include "EngineInterface/SimulationMonitor.h"

#include "Definitions.h"

class SimulationMonitorGpu
	: public SimulationMonitor
{
	Q_OBJECT
public:
	SimulationMonitorGpu(QObject* parent = nullptr) : SimulationMonitor(parent) {}
	virtual ~SimulationMonitorGpu() = default;

	virtual void init(SimulationControllerGpu* controller) = 0;

};
