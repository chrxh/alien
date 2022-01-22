#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _SaveSymbolsDialog
{
public:
    _SaveSymbolsDialog(SimulationController const& simController);

    void process();

    void show();

private:
    SimulationController _simController;
};
