#pragma once

#include "EngineInterface/Definitions.h"

#include "LocationWidgets.h"
#include "ZoneColorPalette.h"

class _SimulationParametersBaseWidgets : public _LocationWidgets
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
