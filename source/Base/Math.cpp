#include "Math.h"

#include <cmath>

float Math::length(RealVector2D const& v)
{
    return sqrt(v.x * v.x + v.y * v.y);
}

float Math::angleOfVector(RealVector2D const& v)
{
    float angle = 0.0;
    auto angleSin = asinf(-v.y / length(v)) * Const::RadToDeg;
    if (v.x >= 0.0) {
        angle = 90.0 - angleSin;
    } else {
        angle = angleSin + 270.0;
    }
    return angle;
}

RealVector2D Math::rotateQuarterCounterClockwise(RealVector2D v)
{
    auto temp = v.x;
    v.x = v.y;
    v.y = -temp;
    return v;
}

RealVector2D Math::unitVectorOfAngle(double angle)
{
    return {sinf(angle * Const::DegToRad), -cosf(angle * Const::DegToRad)};
}

RealMatrix2D Math::calcRotationMatrix(float angle)
{
    RealMatrix2D result;
    result[0][0] = cosf(angle * Const::DegToRad);
    result[1][0] = sinf(angle * Const::DegToRad);
    result[0][1] = -result[1][0];
    result[1][1] = result[0][0];
    return result;
}

RealVector2D operator*(RealMatrix2D const& m, RealVector2D const& v)
{
    return {m[0][0] * v.x + m[0][1] * v.y, m[1][0] * v.x + m[1][1] * v.y};
}
