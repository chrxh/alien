#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

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
