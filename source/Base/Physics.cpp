#include "Physics.h"

#include "Math.h"

RealVector2D
Physics::tangentialVelocity(RealVector2D const& positionFromCenter, RealVector2D const& vel, double angularVel)
{
    return vel - rotateQuarterCounterClockwise(positionFromCenter) * toFloat(angularVel * degToRad);
}

RealVector2D Physics::rotateQuarterCounterClockwise(RealVector2D v)
{
    //90 degree counterclockwise rotation of vector v
    auto temp = v.x;
    v.x = v.y;
    v.y = -temp;
    return v;
}
