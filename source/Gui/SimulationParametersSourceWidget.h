#pragma once

#include <string>

#include "EngineInterface/Definitions.h"
#include "LocationWidget.h"
#include "LayerColorPalette.h"

class _SimulationParametersSourceWidgets : public _LocationWidget
{
public:
    void init(SimulationFacade const& simulationFacade, int orderNumber);
    void process() override;
    std::string getLocationName() override;
    int getOrderNumber() const override;
    void setOrderNumber(int orderNumber) override;

private:
    SimulationFacade _simulationFacade;

    int _orderNumber = 0;
    std::string _sourceName;
};
