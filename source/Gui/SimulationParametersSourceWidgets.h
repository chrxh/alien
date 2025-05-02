#pragma once

#include <string>

#include "EngineInterface/Definitions.h"
#include "LocationWidgets.h"
#include "LayerColorPalette.h"

class _SimulationParametersSourceWidgets : public _LocationWidgets
{
public:
    void init(SimulationFacade const& simulationFacade, int locationIndex);
    void process() override;
    std::string getLocationName() override;
    int getLocationIndex() const override;
    void setLocationIndex(int locationIndex) override;

private:
    SimulationFacade _simulationFacade;

    int _locationIndex = 0;
    std::string _sourceName;
};
