#pragma once

#include <string>

#include <EngineInterface/Definitions.h>

#include "LayerColorPalette.h"
#include "LocationWidget.h"

class _SimulationParametersSourceWidgets : public _LocationWidget
{
public:
    void init(int orderNumber);
    void process(ParametersFilter const& filter) override;
    std::string getLocationName() override;
    int getOrderNumber() const override;
    void setOrderNumber(int orderNumber) override;

private:

    int _orderNumber = 0;
    std::string _sourceName;
};
