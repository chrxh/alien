#pragma once

#include "EngineInterface/SimulationController.h"

class SimulationControllerGpu
	: public SimulationController
{
	Q_OBJECT
public:
	SimulationControllerGpu(QObject* parent = nullptr) : SimulationController(parent) {}
	virtual ~SimulationControllerGpu() = default;
};
