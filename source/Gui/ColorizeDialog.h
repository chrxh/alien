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
    bool _checkColor1 = false;
    bool _checkColor2 = false;
    bool _checkColor3 = false;
    bool _checkColor4 = false;
    bool _checkColor5 = false;
    bool _checkColor6 = false;
    bool _checkColor7 = false;
};
