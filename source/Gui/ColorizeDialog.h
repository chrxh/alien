#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _ColorizeDialog
{
public:
    _ColorizeDialog(SimulationController const& simController);

    void process();

    void show();

private:
    void colorCheckbox(std::string id, uint32_t cellColor, bool& check);

    void onColorize();

    SimulationController _simController;

    bool _show = false;
    bool _checkColors[7] = {false, false, false, false, false, false, false};
};
