#pragma once

#include "Definitions.h"

class EngineInterfaceSettings
{
public:
    static SymbolTable* getDefaultSymbolTable();
	static SimulationParameters getDefaultSimulationParameters();
    static ExecutionParameters getDefaultExecutionParameters();
};

