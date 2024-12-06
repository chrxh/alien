#pragma once

#include "Definitions.h"

class _LocationWidgets
{
public:
    virtual ~_LocationWidgets() = default;

    virtual void process() = 0;
    virtual std::string getLocationName() = 0;
};