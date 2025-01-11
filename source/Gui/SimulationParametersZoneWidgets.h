#pragma once

#include "EngineInterface/SimulationParametersZone.h"
#include "EngineInterface/Definitions.h"

#include "LocationWidgets.h"
#include "ZoneColorPalette.h"

class _SimulationParametersZoneWidgets : public _LocationWidgets
{
public:
    void init(SimulationFacade const& simulationFacade, int locationIndex);
    void process() override;
    std::string getLocationName() override;
    int getLocationIndex() const override;
    void setLocationIndex(int locationIndex) override;

private:
    void setDefaultSpotData(SimulationParametersZone& spot) const;

    SimulationFacade _simulationFacade;

    int _locationIndex = 0;
    ZoneColorPalette _zoneColorPalette;
    uint32_t _backupColor = 0;
    std::vector<std::string> _cellTypeStrings;
    std::string _zoneName;
};
