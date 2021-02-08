#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineGpuKernels/CudaConstants.h"

#include "Definitions.h"

class ModelGpuBuilderFacade
{
public:
	virtual ~ModelGpuBuilderFacade() = default;

	struct Config {
		IntVector2D universeSize;
		SymbolTable* symbolTable;
		SimulationParameters parameters;
	};
	virtual SimulationControllerGpu* buildSimulationController(Config const& config
		, ModelGpuData const& specificData
		, uint timestepAtBeginning = 0) const = 0;
	virtual SimulationAccessGpu* buildSimulationAccess() const = 0;
	virtual SimulationMonitorGpu* buildSimulationMonitor() const = 0;

    virtual CudaConstants getDefaultCudaConstants() const = 0;
};

