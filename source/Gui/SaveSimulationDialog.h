#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _SaveSimulationDialog
{
public:
    _SaveSimulationDialog(SimulationController const& simController);

    void process();

    void show();

private:
    SimulationController _simController;
};
