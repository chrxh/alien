#pragma once

#include "ZoneColorPalette.h"
#include "EngineInterface/Definitions.h"

class SimulationParametersBaseWidgets
{
public:
    void init(SimulationFacade const& simulationFacade);
    void process();

private:
    SimulationFacade _simulationFacade;

    ZoneColorPalette _zoneColorPalette;
    uint32_t _backupColor = 0;
    std::vector<std::string> _cellFunctionStrings;
};
