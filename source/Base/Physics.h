#pragma once

#include "Definitions.h"
#include "DllExport.h"

class Physics
{
public:

    BASE_EXPORT static RealVector2D
    tangentialVelocity(RealVector2D const& positionFromCenter, RealVector2D const& vel, double angularVel);

    BASE_EXPORT static RealVector2D rotateQuarterCounterClockwise(RealVector2D v);
};
