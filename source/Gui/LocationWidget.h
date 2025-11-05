#pragma once

#include <EngineInterface/ParametersFilter.h>

#include "Definitions.h"

class _LocationWidget
{
public:
    virtual ~_LocationWidget() = default;

    virtual void process(ParametersFilter const& filter = ParametersFilter()) = 0;
    virtual std::string getLocationName() = 0;
    virtual int getOrderNumber() const = 0;
    virtual void setOrderNumber(int orderNumber) = 0;
};
