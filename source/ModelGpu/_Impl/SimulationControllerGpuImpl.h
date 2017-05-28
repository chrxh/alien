#pragma once

#include "Model/SimulationController.h"
#include "DefinitionsImpl.h"

class SimulationControllerGpuImpl
	: public SimulationController
{
	Q_OBJECT
public:
	SimulationControllerGpuImpl(QObject* parent = nullptr) : SimulationController(parent) {}
	virtual ~SimulationControllerGpuImpl() = default;

	virtual void init(SimulationContextApi* context) override;
	virtual void setRun(bool run) override;
	virtual void calculateSingleTimestep() override;
	virtual SimulationContextApi* getContext() const override;

private:
	SimulationContextGpuImpl *_context = nullptr;
};