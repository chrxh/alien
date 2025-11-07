#pragma once

#include <EngineInterface/Definitions.h>

#include "LayerColorPalette.h"
#include "LocationWidget.h"

class _SimulationParameterLayerWidget : public _LocationWidget
{
public:
    void init(SimulationFacade const& simulationFacade, int orderNumber);
    void process(ParametersFilter const& filter) override;
    std::string getLocationName() override;
    int getOrderNumber() const override;
    void setOrderNumber(int orderNumber) override;

private:

    int _orderNumber = 0;
    std::vector<std::string> _cellTypeStrings;
    std::string _layerName;
};
