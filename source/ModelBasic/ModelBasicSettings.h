#pragma once

#include "Definitions.h"

class ModelBasicSettings
{
public:
    static SymbolTable* getDefaultSymbolTable();
	static SimulationParameters getDefaultSimulationParameters();
    static ExecutionParameters getDefaultExecutionParameters();
};

