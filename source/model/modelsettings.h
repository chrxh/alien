#ifndef MODELSETTINGS_H
#define MODELSETTINGS_H

#include "Definitions.h"

const qreal ALIEN_PRECISION = 0.0000001;

class ModelData
{
public:
    static void loadDefaultSymbolTable (SymbolTable* meta);
	static void loadDefaultSimulationParameters(SimulationParameters* parameters);
};

#endif // MODELSETTINGS_H
