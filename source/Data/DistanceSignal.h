#pragma once

#include "CellTypeConstants.h"
#include "Definitions.h"

class DistanceSignal
{
public:
    HOST_DEVICE static float encode(float distance) { return 1.0f / (1.0f + distance / Const::DistanceAtHalfSignal); }
};
