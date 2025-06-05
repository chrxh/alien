#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParametersSpecification.h"

#include "LocationWidget.h"

class _SimulationParametersBaseWidget : public _LocationWidget
{
public:
    void init(SimulationFacade const& simulationFacade);
    void process() override;
    std::string getLocationName() override;
    int getOrderNumber() const override;
    void setOrderNumber(int orderNumber) override;

private:
    SimulationFacade _simulationFacade;
};
