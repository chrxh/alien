#ifndef MODELSETTINGS_H
#define MODELSETTINGS_H

#include "Definitions.h"

const qreal ALIEN_PRECISION = 0.0000001;

class ModelSettings
{
public:
    static SymbolTable* loadDefaultSymbolTable();
	static SimulationParameters* loadDefaultSimulationParameters();
};

#endif // MODELSETTINGS_H
