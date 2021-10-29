#include "Math.h"

#include <cmath>

double Math::length(RealVector2D const& v)
{
    return sqrt(v.x * v.x + v.y * v.y);
}

double Math::angleOfVector(RealVector2D const& v)
{
    double angle(0.0);
    double angleSin(asin(-v.y / length(v)) * radToDeg);
    if (v.x >= 0.0) {
        angle = 90.0 - angleSin;
    } else {
        angle = angleSin + 270.0;
    }
    return angle;
}