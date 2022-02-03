#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _OpenSymbolsDialog
{
public:
    _OpenSymbolsDialog(SimulationController const& simController);
    ~_OpenSymbolsDialog();

    void process();

    void show();

private:
    SimulationController _simController;

    std::string _startingPath;
};
