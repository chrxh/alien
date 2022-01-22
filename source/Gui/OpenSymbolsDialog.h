#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _OpenSymbolsDialog
{
public:
    _OpenSymbolsDialog(SimulationController const& simController);

    void process();

    void show();

private:
    SimulationController _simController;
};
