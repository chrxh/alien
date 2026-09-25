#pragma once

#include <Data/SimulationParametersSpecification.h>

#include <EngineInterface/Definitions.h>

#include "LocationWidget.h"

class _SimulationParametersBaseWidget : public _LocationWidget
{
public:
    void init();
    void process(ParametersFilter const& filter) override;
    std::string getLocationName() override;
    int getOrderNumber() const override;
    void setOrderNumber(int orderNumber) override;
};
