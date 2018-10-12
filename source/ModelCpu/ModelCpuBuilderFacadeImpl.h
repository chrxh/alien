#pragma once

#include "Definitions.h"
#include "ModelCpuBuilderFacade.h"

class ModelCpuBuilderFacadeImpl
	: public ModelCpuBuilderFacade
{
public:
	~ModelCpuBuilderFacadeImpl() = default;

	SimulationControllerCpu* buildSimulationController(Config const& config
		, ModelCpuData const& specificData
		, uint timestepAtBeginning) const override;
	SimulationAccessCpu* buildSimulationAccess() const override;
	SimulationMonitor* buildSimulationMonitor() const override;

private:
	Unit* buildSimulationUnit(IntVector2D gridPos, SimulationContextImpl* context) const;
};
