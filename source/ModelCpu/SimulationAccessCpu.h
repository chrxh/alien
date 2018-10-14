#pragma once

#include "ModelBasic/SimulationAccess.h"
#include "Definitions.h"

class SimulationAccessCpu
	: public SimulationAccess
{
	Q_OBJECT
public:
	SimulationAccessCpu(QObject* parent = nullptr) : SimulationAccess(parent) {}
	virtual ~SimulationAccessCpu() = default;

	virtual void init(SimulationControllerCpu* controller) = 0;
};
