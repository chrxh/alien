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
    void checkbox(std::string id, uint64_t cellColor, bool& check);

    void onColorize();

    SimulationController _simController;

    bool _show = false;
    bool _checkColors[7] = {false, false, false, false, false, false, false};
};
