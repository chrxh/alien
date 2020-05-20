#pragma once

#include "Definitions.h"

class MODELBASIC_EXPORT ModelSettings
{
public:
    static SymbolTable* getDefaultSymbolTable();
	static SimulationParameters getDefaultSimulationParameters();
    static ExecutionParameters getDefaultExecutionParameters();
};

