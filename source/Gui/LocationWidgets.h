#pragma once

#include "Definitions.h"

class _LocationWidgets
{
public:
    virtual ~_LocationWidgets() = default;

    virtual void process() = 0;
    virtual std::string getLocationName() = 0;
    virtual int getOrderNumber() const = 0;
    virtual void setOrderNumber(int orderNumber) = 0;
};