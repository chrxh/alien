#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _SaveSimulationDialog
{
public:
    _SaveSimulationDialog(SimulationController const& simController);
    ~_SaveSimulationDialog();

    void process();

    void show();

private:
    SimulationController _simController;
    std::string _startingPath;
};
