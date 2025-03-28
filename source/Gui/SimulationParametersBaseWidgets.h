#pragma once

#include "EngineInterface/Definitions.h"

#include "LocationWidgets.h"
#include "ZoneColorPalette.h"
#include "EngineInterface/SimulationParametersSpecification.h"

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
};
