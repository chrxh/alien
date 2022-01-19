#include "Physics.h"

#include "Math.h"

RealVector2D Physics::tangentialVelocity(RealVector2D const& positionFromCenter, RealVector2D const& vel, double angularVel)
{
    return vel - Math::rotateQuarterCounterClockwise(positionFromCenter) * toFloat(angularVel * Const::DegToRad);
}
