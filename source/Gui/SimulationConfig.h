#pragma once

#include "ModelGpu/CudaConstants.h"

#include "Definitions.h"

class _SimulationConfig
{
public:
	virtual ~_SimulationConfig() = default;

	enum class ValidationResult {
		Ok,
		Error
	};

	virtual ValidationResult validate(string& errorMsg) const = 0;

	IntVector2D universeSize;
	SymbolTable* symbolTable;
	SimulationParameters parameters;
};


class _SimulationConfigGpu
	: public _SimulationConfig
{
public:
	virtual ValidationResult validate(string& errorMsg) const override;

    CudaConstants cudaConstants;
};
