#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _ColorizeDialog
{
public:
    _ColorizeDialog(SimulationController const& simController);

    void process();

    void show();

private:
    void onColorize();

    SimulationController _simController;

    bool _show = false;
};
