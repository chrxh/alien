#include "Math.h"

#include <cmath>

float Math::length(RealVector2D const& v)
{
    return sqrt(v.x * v.x + v.y * v.y);
}

float Math::angleOfVector(RealVector2D const& v)
{
    float angle = 0.0;
    auto angleSin = asinf(-v.y / length(v)) * radToDeg;
    if (v.x >= 0.0) {
        angle = 90.0 - angleSin;
    } else {
        angle = angleSin + 270.0;
    }
    return angle;
}

RealVector2D Math::unitVectorOfAngle(double angle)
{
    return {sinf(angle * degToRad), -cosf(angle * degToRad)};
}
