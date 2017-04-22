#ifndef CONFIG_H
#define CONFIG_H

#include "definitions.h"

const qreal ALIEN_PRECISION = 0.0000001;

class Metadata
{
public:
    static void loadDefaultSymbolTable (SymbolTable* meta);
	static void loadDefaultSimulationParameters(SimulationParameters* parameters);
};

#endif // CONFIG_H
