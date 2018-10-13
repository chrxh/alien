#pragma once

#include "ModelBasic/SimulationController.h"

class SimulationControllerGpu
	: public SimulationController
{
	Q_OBJECT
public:
	SimulationControllerGpu(QObject* parent = nullptr) : SimulationController(parent) {}
	virtual ~SimulationControllerGpu() = default;
};
