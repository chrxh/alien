#pragma once

#include "ModelGpu/ModelGpuBuilderFacade.h"

class ModelGpuBuilderFacadeImpl
	: public ModelGpuBuilderFacade
{
public:
	virtual ~ModelGpuBuilderFacadeImpl() = default;

    SimulationControllerGpu* buildSimulationController(
        Config const& config,
        ModelGpuData const& specificData,
        uint timestepAtBeginning) const override;
    SimulationAccessGpu* buildSimulationAccess() const override;
	SimulationMonitorGpu* buildSimulationMonitor() const override;

    CudaConstants getDefaultCudaConstants() const override;

};
