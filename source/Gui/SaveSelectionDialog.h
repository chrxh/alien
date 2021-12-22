#pragma once

#include "Definitions.h"
#include "EngineImpl/Definitions.h"

class _SaveSelectionDialog
{
public:
    _SaveSelectionDialog(SimulationController const& simController);

    void process();

    void show(bool includeClusters);
private:
    SimulationController _simController;
    bool _includeClusters = true;
};
