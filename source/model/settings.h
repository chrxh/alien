#ifndef SETTINGS_H
#define SETTINGS_H

#include "Definitions.h"

const qreal ALIEN_PRECISION = 0.0000001;

class ModelSettings
{
public:
    static SymbolTable* loadDefaultSymbolTable();
	static SimulationParameters* loadDefaultSimulationParameters();
};

#endif // SETTINGS_H
