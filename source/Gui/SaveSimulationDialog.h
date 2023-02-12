#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _SaveSimulationDialog
{
public:
    _SaveSimulationDialog(SimulationController const& simController, Viewport const& viewport);
    ~_SaveSimulationDialog();

    void process();

    void show();

private:
    SimulationController _simController;
    Viewport _viewport;
    std::string _startingPath;
};
