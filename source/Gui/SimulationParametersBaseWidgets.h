#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParametersSpecification.h"

#include "LocationWidgets.h"

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
