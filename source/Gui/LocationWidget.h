#pragma once

#include "Definitions.h"

class _LocationWidget
{
public:
    virtual ~_LocationWidget() = default;

    virtual void process() = 0;
    virtual std::string getLocationName() = 0;
    virtual int getOrderNumber() const = 0;
    virtual void setOrderNumber(int orderNumber) = 0;
};