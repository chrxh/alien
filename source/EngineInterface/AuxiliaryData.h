#pragma once

#include "Base/Definitions.h"
#include "Settings.h"

struct AuxiliaryData
{
    uint64_t timestep;
    float zoom;
    RealVector2D center;
    Settings settings;
};
