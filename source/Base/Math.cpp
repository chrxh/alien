#include "Math.h"

#include <cmath>

float Math::length(RealVector2D const& v)
{
    return sqrt(v.x * v.x + v.y * v.y);
}

float Math::angleOfVector(RealVector2D const& v)
{
    auto vLength = length(v);
    if (vLength < NEAR_ZERO) {
        return 0;
    }
    float angle;
    auto angleSin = asinf(-v.y / vLength) * Const::RadToDeg;
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

RealVector2D Math::unitVectorOfAngle(float angle)
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

bool Math::crossing(
    RealVector2D const& segmentStart,
    RealVector2D const& segmentEnd,
    RealVector2D const& otherSegmentStart,
    RealVector2D const& otherSegmentEnd)
{
    auto const& p1 = segmentStart;
    auto v1 = segmentEnd - segmentStart;
    auto const& p2 = otherSegmentStart;
    auto v2 = otherSegmentEnd - otherSegmentStart;

    auto divisor = v2.x * v1.y - v2.y * v1.x;
    if (abs(divisor) < NEAR_ZERO) {
        return false;
    }
    auto mue = (v1.x * (p2.y - p1.y) - v1.y * (p2.x - p1.x)) / divisor;
    if (mue < -NEAR_ZERO || mue > 1 + NEAR_ZERO) {
        return false;
    }

    float lambda;
    if (abs(v1.x) > NEAR_ZERO) {
        lambda = (p2.x - p1.x + mue * v2.x) / v1.x;
    } else if (abs(v1.y) > NEAR_ZERO) {
        lambda = (p2.y - p1.y + mue * v2.y) / v1.y;
    } else {
        return false;
    }

    return lambda >= NEAR_ZERO && lambda <= 1 - NEAR_ZERO;
}

float Math::sigmoid(float x)
{
    return 2.0f / (1.0f + expf(-x)) - 1.0f;
}

float Math::binaryStep(float x)
{
    return x >= NEAR_ZERO ? 1.0f : 0.0f;
}

float Math::gaussian(float x)
{
    return expf(-2 * x * x);
}

RealVector2D operator*(RealMatrix2D const& m, RealVector2D const& v)
{
    return {m[0][0] * v.x + m[0][1] * v.y, m[1][0] * v.x + m[1][1] * v.y};
}
