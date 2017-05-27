#ifndef MODEL_SETTINGS_H
#define MODEL_SETTINGS_H

#include "Definitions.h"

const qreal ALIEN_PRECISION = 0.0000001;

class MODEL_EXPORT ModelSettings
{
public:
    static SymbolTable* loadDefaultSymbolTable();
	static SimulationParameters* loadDefaultSimulationParameters();
};

#endif // MODEL_SETTINGS_H
