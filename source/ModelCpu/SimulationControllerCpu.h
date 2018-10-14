#pragma once

#include "ModelBasic/SimulationController.h"

class SimulationControllerCpu
	: public SimulationController
{
	Q_OBJECT
public:
	SimulationControllerCpu(QObject* parent = nullptr) : SimulationController(parent) {}
	virtual ~SimulationControllerCpu() = default;
};