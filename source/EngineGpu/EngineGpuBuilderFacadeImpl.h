#pragma once

#include "EngineGpuBuilderFacade.h"

class EngineGpuBuilderFacadeImpl
	: public EngineGpuBuilderFacade
{
public:
	virtual ~EngineGpuBuilderFacadeImpl() = default;

    SimulationControllerGpu* buildSimulationController(
        Config const& config,
        EngineGpuData const& specificData,
        uint timestepAtBeginning) const override;
    SimulationAccessGpu* buildSimulationAccess() const override;
	SimulationMonitorGpu* buildSimulationMonitor() const override;

    CudaConstants getDefaultCudaConstants() const override;

};
