#pragma once

#include "ModelGpu/ModelGpuBuilderFacade.h"

class ModelGpuBuilderFacadeImpl
	: public ModelGpuBuilderFacade
{
public:
	virtual ~ModelGpuBuilderFacadeImpl() = default;

	virtual SimulationControllerGpu* buildSimulationController(Config const& config
		, ModelGpuData const& specificData
		, uint timestepAtBeginning) const override;
	virtual SimulationAccessGpu* buildSimulationAccess() const override;
	virtual SimulationMonitorGpu* buildSimulationMonitor() const override;
};
