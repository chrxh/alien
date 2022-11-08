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

RealVector2D Math::rotateClockwise(RealVector2D const& v, float angle)
{
    return calcRotationMatrix(angle) * v;
}

void Math::normalize(RealVector2D& v)
{
    float l = length(v);
    if (l > 0.0001f) {
        v = {v.x / l, v.y / l};
    } else {
        v = {1.0f, 0.0f};
    }
}

float Math::subtractAngle(float angleMinuend, float angleSubtrahend)
{
    auto angleDiff = angleMinuend - angleSubtrahend;
    if (angleDiff > 360.0f) {
        angleDiff -= 360.0f;
    }
    if (angleDiff < 0.0f) {
        angleDiff += 360.0f;
    }
    return angleDiff;
}

bool Math::isAngleInBetween(float angle1, float angle2, float angleBetweenCandidate)
{
    if (angle1 == angle2 && angle1 != angleBetweenCandidate) {
        return false;
    }
    if (angleBetweenCandidate < angle1) {
        angleBetweenCandidate += 360.0f;
        angle2 += 360.0f;
    }
    if (angle2 < angleBetweenCandidate) {
        angle2 += 360.0f;
    }
    return angle2 - angle1 < 360.0f;
}

RealVector2D operator*(RealMatrix2D const& m, RealVector2D const& v)
{
    return {m[0][0] * v.x + m[0][1] * v.y, m[1][0] * v.x + m[1][1] * v.y};
}
