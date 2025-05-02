#pragma once

#include "EngineInterface/Definitions.h"

#include "LocationWidgets.h"
#include "LayerColorPalette.h"

class _SimulationParametersLayerWidgets : public _LocationWidgets
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
    std::vector<std::string> _cellTypeStrings;
    std::string _layerName;
};
