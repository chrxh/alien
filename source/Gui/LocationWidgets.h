#pragma once

#include "Definitions.h"

class _LocationWidgets
{
public:
    virtual ~_LocationWidgets() = default;

    virtual void process() = 0;
    virtual std::string getLocationName() = 0;
    virtual int getLocationIndex() const = 0;
    virtual void setLocationIndex(int locationIndex) = 0;
};