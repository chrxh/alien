#pragma once

#include "EngineGpuKernels/CudaConstants.h"

#include "Definitions.h"

class _SimulationConfig
{
public:
	virtual ~_SimulationConfig() = default;

	enum class ValidationResult {
		Ok,
		Error
	};

	virtual ValidationResult validate(string& errorMsg) const;

	IntVector2D universeSize;
	SymbolTable* symbolTable;
	SimulationParameters parameters;
    CudaConstants cudaConstants;
};
