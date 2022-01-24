#pragma once

#include "Definitions.h"

class Physics
{
public:

    static RealVector2D
    tangentialVelocity(RealVector2D const& positionFromCenter, RealVector2D const& vel, double angularVel);

    static RealVector2D rotateQuarterCounterClockwise(RealVector2D v);
};
