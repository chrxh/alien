#pragma once

#include "EngineInterface/SimulationParametersZone.h"
#include "EngineInterface/Definitions.h"

#include "LocationWidgets.h"
#include "ZoneColorPalette.h"

class _SimulationParametersZoneWidgets : public _LocationWidgets
{
public:
    void init(SimulationFacade const& simulationFacade, int zoneIndex);
    void process() override;
    std::string getLocationName() override;

private:
    void setDefaultSpotData(SimulationParametersZone& spot) const;


    SimulationFacade _simulationFacade;

    int _zoneIndex = 0;
    ZoneColorPalette _zoneColorPalette;
    uint32_t _backupColor = 0;
    std::vector<std::string> _cellFunctionStrings;
    std::string _zoneName;
};
