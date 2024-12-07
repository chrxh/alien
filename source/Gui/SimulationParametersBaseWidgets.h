#pragma once

#include "EngineInterface/Definitions.h"

#include "LocationWidgets.h"
#include "ZoneColorPalette.h"

class _SimulationParametersBaseWidgets : public _LocationWidgets
{
public:
    void init(SimulationFacade const& simulationFacade);
    void process() override;
    std::string getLocationName() override;
    int getLocationIndex() const override;
    void setLocationIndex(int locationIndex) override;

private:
    SimulationFacade _simulationFacade;

    ZoneColorPalette _zoneColorPalette;
    uint32_t _backupColor = 0;
    std::vector<std::string> _cellFunctionStrings;
};
