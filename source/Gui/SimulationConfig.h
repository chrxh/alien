#pragma once

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


class _SimulationConfigCpu
	: public _SimulationConfig
{
public:
	virtual ValidationResult validate(string& errorMsg) const override;

	uint maxThreads = 0;
	IntVector2D gridSize;
};

class _SimulationConfigGpu
	: public _SimulationConfig
{
public:
	virtual ValidationResult validate(string& errorMsg) const override;

    uint numBlocks = 0;
    uint numThreadsPerBlock = 0;
    uint maxClusters = 0;
    uint maxCells = 0;
    uint maxTokens = 0;
    uint maxParticles = 0;
    uint dynamicMemorySize = 0;
};
