#pragma once

#include "EngineInterface/SimulationAccess.h"
#include "Definitions.h"

class SimulationAccessGpu
	: public SimulationAccess
{
	Q_OBJECT
public:
	SimulationAccessGpu(QObject* parent = nullptr) : SimulationAccess(parent) {}
	virtual ~SimulationAccessGpu() = default;

	virtual void init(SimulationControllerGpu* controller) = 0;
};
