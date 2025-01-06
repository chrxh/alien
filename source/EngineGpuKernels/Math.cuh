#pragma once

#include "EngineInterface/CellFunctionConstants.h"

#include "Base.cuh"

namespace Const
{
    static float constexpr PI = 3.1415926535897932384626433832795f;
    static float constexpr DEG_TO_RAD = PI / 180.0f;
    static float constexpr RAD_TO_DEG = 180.0f / PI;
}

class Math
{
public:

    using Matrix = float[2][2];

    __inline__ __device__ static void rotationMatrix(float angle, Matrix& rotMatrix);
    __inline__ __device__ static void inverseRotationMatrix(float angle, Matrix& rotMatrix);
    __inline__ __device__ static float2 applyMatrix(float2 const& vec, Matrix const& matrix);
    __inline__ __device__ static void angleCorrection(float& angle);
    __inline__ __device__ static void angleCorrection(int& angle);
    __inline__ __device__ static bool isInBetweenModulo(float value1, float value2, float candidate, float size);
    __inline__ __device__ static bool isAngleInBetween(float angle1, float angle2, float angleBetweenCandidate);
    __inline__ __device__ static void rotateQuarterClockwise(float2& v);
    __inline__ __device__ static void rotateQuarterCounterClockwise(float2& v);
    __inline__ __device__ static float angleOfVector(float2 const& v);  //0 DEG corresponds to (0,-1)
    __inline__ __device__ static float2 unitVectorOfAngle(float angle);
    __inline__ __device__ static void normalize(float2& vec);
    __inline__ __device__ static float2 normalized(float2 vec);
    __inline__ __device__ static float dot(float2 const& p, float2 const& q);
    __inline__ __device__ static float2 crossProdProjected(float3 const& p, float3 const& q);
    __inline__ __device__ static float length(float2 const& v);
    __inline__ __device__ static float lengthMax(float2 const& v);
    __inline__ __device__ static float length(int2 const& v);
    __inline__ __device__ static float lengthSquared(float2 const& v);
    __inline__ __device__ static float2 rotateClockwise(float2 const& v, float angle);
    __inline__ __device__ static float subtractAngle(float angleMinuend, float angleSubtrahend);
    __inline__ __device__ static float calcDistanceToLineSegment(float2 const& startSegment, float2 const& endSegment, float2 const& pos, float boundary = 0);
    __inline__ __device__ static float alignAngle(float angle, ConstructorAngleAlignment alignment);
    __inline__ __device__ static float alignAngleOnBoundaries(float angle, float maxAngle, ConstructorAngleAlignment alignment);
    __inline__ __device__ static bool crossing(float2 const& segmentStart, float2 const& segmentEnd, float2 const& otherSegmentStart, float2 const& otherSegmentEnd);
    __inline__ __device__ static float modulo(float value, float size);
};

__inline__ __device__ __host__ float2 operator+(float2 const& p, float2 const& q)
{
    return{ p.x + q.x, p.y + q.y };
}

__inline__ __device__ __host__ float3 operator+(float3 const& p, float3 const& q)
{
    return {p.x + q.x, p.y + q.y, p.z + q.z};
}

__inline__ __device__ __host__ float2 operator-(float2 const& p, float2 const& q)
{
    return{ p.x - q.x, p.y - q.y };
}

__inline__ __device__ __host__ float2 operator-(float2 const& p, int2 const& q)
{
    return{ p.x - q.x, p.y - q.y };
}

__inline__ __device__ __host__ int2 operator-(int2 const& p, int2 const& q)
{
    return{ p.x - q.x, p.y - q.y };
}

__inline__ __device__ __host__ float2 operator*(float2 const& p, float m)
{
    return{ p.x * m, p.y * m };
}

__inline__ __device__ __host__ float3 operator*(float3 const& p, float m)
{
    return {p.x * m, p.y * m, p.z * m};
}

__inline__ __device__ __host__ float2 operator/(float2 const& p, float m)
{
    return{ p.x / m, p.y / m };
}

__inline__ __device__ __host__ float3 operator/(float3 const& p, float m)
{
    return {p.x / m, p.y / m, p.z / m};
}

__inline__ __device__ __host__ bool operator==(int2 const& p, int2 const& q)
{
    return p.x == q.x && p.y == q.y;
}

__inline__ __device__ __host__ void operator*=(float2& p, float const& q)
{
    p.x *= q;
    p.y *= q;
}

__inline__ __device__ __host__ void operator+=(float2& p, float2 const& q)
{
    p.x += q.x;
    p.y += q.y;
}

__inline__ __device__ __host__ void operator*=(float3& p, float const& q)
{
    p.x *= q;
    p.y *= q;
    p.z *= q;
}

__inline__ __device__ __host__ void operator+=(float3& p, float3 const& q)
{
    p.x += q.x;
    p.y += q.y;
    p.z += q.z;
}

__inline__ __device__ __host__ void operator-=(float2& p, float2 const& q)
{
    p.x -= q.x;
    p.y -= q.y;
}

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ float2 Math::unitVectorOfAngle(float angle)
{
    angle *= Const::DEG_TO_RAD;
    return{ sinf(angle), -cosf(angle) };
}

__inline__ __device__ float Math::angleOfVector(float2 const & v)
{
    if (length(v) < NEAR_ZERO) {
        return 0;
    }

    auto normalizedVy = -v.y / length(v);
    normalizedVy = max(-1.0f, min(1.0f, normalizedVy));
    float angleSin = asinf(normalizedVy) * Const::RAD_TO_DEG;
    if (v.x >= 0.0f) {
        return 90.0f - angleSin;
    }
    else {
        return angleSin + 270.0f;
    }
}

__inline__ __device__ void Math::rotateQuarterClockwise(float2& v)
{
    float temp = v.x;
    v.x = -v.y;
    v.y = temp;
}

__device__ __inline__ void Math::rotateQuarterCounterClockwise(float2 &v)
{
    float temp = v.x;
    v.x = v.y;
    v.y = -temp;
}

__inline__ __device__ void Math::rotationMatrix(float angle, Matrix& rotMatrix)
{
    float sinAngle = __sinf(angle*Const::DEG_TO_RAD);
    float cosAngle = __cosf(angle*Const::DEG_TO_RAD);
    rotMatrix[0][0] = cosAngle;
    rotMatrix[0][1] = -sinAngle;
    rotMatrix[1][0] = sinAngle;
    rotMatrix[1][1] = cosAngle;
}

__inline__ __device__ void Math::inverseRotationMatrix(float angle, Matrix& rotMatrix)
{
    float sinAngle = __sinf(angle*Const::DEG_TO_RAD);
    float cosAngle = __cosf(angle*Const::DEG_TO_RAD);
    rotMatrix[0][0] = cosAngle;
    rotMatrix[0][1] = sinAngle;
    rotMatrix[1][0] = -sinAngle;
    rotMatrix[1][1] = cosAngle;
}

__inline__ __device__ float2 Math::applyMatrix(float2 const & vec, Matrix const & matrix)
{
    return{ vec.x * matrix[0][0] + vec.y * matrix[0][1],  vec.x * matrix[1][0] + vec.y * matrix[1][1] };
}

__inline__ __device__ void Math::angleCorrection(int &angle)
{
    angle = ((angle % 360) + 360) % 360;
}

__inline__ __device__ bool Math::isInBetweenModulo(float value1, float value2, float candidate, float size)
{
    if (value2 - value1 >= size) {
        return true;
    }
    auto valueMod1 = modulo(value1, size);
    auto valueMod2 = modulo(value2, size);
    auto candidateMod = modulo(candidate, size);

    if (valueMod1 == valueMod2 && valueMod1 != candidateMod) {
        return false;
    }
    if (candidateMod < valueMod1) {
        candidateMod += size;
        valueMod2 += size;
    }
    if (valueMod2 < candidateMod) {
        valueMod2 += size;
    }
    return valueMod2 - valueMod1 < size;
}

__inline__ __device__ bool Math::isAngleInBetween(float angle1, float angle2, float angleBetweenCandidate)
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

__inline__ __device__ void Math::angleCorrection(float &angle)
{
    int intPart = (int)angle;
    float fracPart = angle - intPart;
    angleCorrection(intPart);
    angle = (float)intPart + fracPart;
}

__device__ __inline__ void Math::normalize(float2 &vec)
{
    float length = sqrt(vec.x*vec.x + vec.y*vec.y);
    if (length > NEAR_ZERO) {
        vec = { vec.x / length, vec.y / length };
    }
    else {
        vec = { 1.0f, 0.0f };
    }
}

__device__ __inline__ float2 Math::normalized(float2 vec)
{
    normalize(vec);
    return vec;
}

__device__ __inline__ float Math::dot(float2 const &p, float2 const &q)
{
    return p.x*q.x + p.y*q.y;
}

__inline__ __device__ float2 Math::crossProdProjected(float3 const& p, float3 const& q)
{
    return {p.y * q.z - p.z * q.y, p.z * q.x - p.x * q.z};
}

__device__ __inline__ float Math::length(float2 const & v)
{
    return sqrt(v.x * v.x + v.y * v.y);
}

__device__ __inline__ float Math::lengthMax(float2 const& v)
{
    return max(abs(v.x), abs(v.y));
}

__device__ __inline__ float Math::length(int2 const & v)
{
    return sqrt(static_cast<float>(v.x * v.x + v.y * v.y));
}

__device__ __inline__ float Math::lengthSquared(float2 const & v)
{
    return v.x * v.x + v.y * v.y;
}

__inline__ __device__ float2 Math::rotateClockwise(float2 const & v, float angle)
{
    Matrix rotMatrix;
    rotationMatrix(angle, rotMatrix);
    return applyMatrix(v, rotMatrix);
}

__inline__ __device__ float Math::subtractAngle(float angleMinuend, float angleSubtrahend)
{
    auto angleDiff = angleMinuend - angleSubtrahend;
    if (angleDiff > 360.0f) {
        angleDiff -= 360.0f;
    }
    if (angleDiff < 0.0) {
        angleDiff += 360.0f;
    }
    return angleDiff;
}

__inline__ __device__ float
Math::calcDistanceToLineSegment(float2 const& startSegment, float2 const& endSegment, float2 const& pos, float boundary)
{
    auto const relPos = pos - startSegment;
    auto segmentDirection = endSegment - startSegment;
    if (length(segmentDirection) < NEAR_ZERO) {
        return boundary + 1.0f;
    }

    auto const segmentLength = length(segmentDirection);
    segmentDirection = segmentDirection / segmentLength;
    auto normal = segmentDirection;
    rotateQuarterCounterClockwise(normal);
    auto const signedDistanceFromLine = dot(relPos, normal);
    if (abs(signedDistanceFromLine) > boundary) {
        return boundary + 1.0f;
    }

    auto const signedDistanceFromStart = dot(relPos, segmentDirection);
    if (signedDistanceFromStart < 0 || signedDistanceFromStart > segmentLength) {
        return boundary + 1.0f;
    }

    return abs(signedDistanceFromLine);
}

__inline__ __device__ float Math::alignAngle(float angle, ConstructorAngleAlignment alignment)
{
    if (ConstructorAngleAlignment_None == alignment) {
        return angle;
    }
    float unitAngle = 360.0f / (alignment + 1);
    float factor = angle / unitAngle + 0.5f;
    factor = floorf(factor);
    return factor * unitAngle;
}

__inline__ __device__ float Math::alignAngleOnBoundaries(float angle, float maxAngle, ConstructorAngleAlignment alignment)
{
    if (alignment != ConstructorAngleAlignment_None) {
        auto angleUnit = 360.0f / (alignment + 1);
        if (angle < NEAR_ZERO && angleUnit < maxAngle - NEAR_ZERO) {
            angle = angleUnit;
        }
        if (angle > maxAngle - NEAR_ZERO && maxAngle - angleUnit > NEAR_ZERO) {
            angle = maxAngle - angleUnit;
        }
    }
    return angle;
}

__inline__ __device__ bool Math::crossing(float2 const& segmentStart, float2 const& segmentEnd, float2 const& otherSegmentStart, float2 const& otherSegmentEnd)
{
    if ((segmentStart.x == otherSegmentStart.x && segmentStart.y == otherSegmentStart.y)
        || (segmentStart.x == otherSegmentEnd.x && segmentStart.y == otherSegmentEnd.y) || (segmentEnd.x == otherSegmentStart.x && segmentEnd.y == otherSegmentStart.y)
        || (segmentEnd.x == otherSegmentEnd.x && segmentEnd.y == otherSegmentEnd.y)) {
        return false;
    }

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

__inline__ __device__ float Math::modulo(float value, float size)
{
    return fmodf(fmodf(value, size) + size, size);
}
