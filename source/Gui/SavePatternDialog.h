#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _SavePatternDialog
{
public:
    _SavePatternDialog(SimulationController const& simController);
    ~_SavePatternDialog();

    void process();

    void show(bool includeClusters);
private:
    SimulationController _simController;
    bool _includeClusters = true;
    std::string _startingPath;
};
