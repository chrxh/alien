#pragma once

#include "Base.cuh"

#define PI 3.1415926535897932384626433832795
#define DEG_TO_RAD PI/180.0
#define RAD_TO_DEG 180.0/PI

class Math
{
public:
    using Matrix = float[2][2];

    __inline__ __device__ static void rotationMatrix(float angle, Matrix& rotMatrix);
    __inline__ __device__ static void inverseRotationMatrix(float angle, Matrix& rotMatrix);
    __inline__ __device__ static float2 applyMatrix(float2 const& vec, Matrix const& matrix);
    __inline__ __device__ static void angleCorrection(float& angle);
    __inline__ __device__ static void rotateQuarterCounterClockwise(float2& v);
    __inline__ __device__ static float angleOfVector(float2 const& v);   //0 DEG corresponds to (0,-1)
    __inline__ __device__ static float2 unitVectorOfAngle(float angle);
    __inline__ __device__ static void normalize(float2& vec);
    __inline__ __device__ static float2 normalized(float2 vec);
    __inline__ __device__ static float dot(float2 const& p, float2 const& q);
    __inline__ __host__ __device__ static float length(float2 const& v);
    __inline__ __host__ __device__ static float lengthSquared(float2 const& v);
    __inline__ __device__ static float2 rotateClockwise(float2 const& v, float angle);
    __inline__ __device__ static float subtractAngle(float angleMinuend, float angleSubtrahend);
    __inline__ __device__ static float
    calcDistanceToLineSegment(float2 const& startSegment, float2 const& endSegment, float2 const& pos, int const& boundary = 0);


private:
    __inline__ __device__ static void angleCorrection(int &angle);

};

__inline__ __device__ float2 operator+(float2 const& p, float2 const& q)
{
    return{ p.x + q.x, p.y + q.y };
}

__inline__ __device__ float2 operator-(float2 const& p, float2 const& q)
{
    return{ p.x - q.x, p.y - q.y };
}

__inline__ __device__ int2 operator-(int2 const& p, int2 const& q)
{
    return{ p.x - q.x, p.y - q.y };
}

__inline__ __device__ float2 operator*(float2 const& p, float m)
{
    return{ p.x * m, p.y * m };
}

__inline__ __device__ float2 operator/(float2 const& p, float m)
{
    return{ p.x / m, p.y / m };
}

__inline__ __device__ bool operator==(int2 const& p, int2 const& q)
{
    return p.x == q.x && p.y == q.y;
}

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ float2 Math::unitVectorOfAngle(float angle)
{
    angle *= DEG_TO_RAD;
    return{ sinf(angle), -cos(angle) };
}

__inline__ __device__ float Math::angleOfVector(float2 const & v)
{
    if (length(v) < FP_PRECISION) {
        return 0;
    }

    float angleSin = asinf(-v.y / length(v)) * RAD_TO_DEG;
    if (v.x >= 0.0f) {
        return 90.0f - angleSin;
    }
    else {
        return angleSin + 270.0f;
    }
}

__device__ __inline__ void Math::rotateQuarterCounterClockwise(float2 &v)
{
    float temp = v.x;
    v.x = v.y;
    v.y = -temp;
}

__inline__ __device__ void Math::rotationMatrix(float angle, Matrix& rotMatrix)
{
    float sinAngle = __sinf(angle*DEG_TO_RAD);
    float cosAngle = __cosf(angle*DEG_TO_RAD);
    rotMatrix[0][0] = cosAngle;
    rotMatrix[0][1] = -sinAngle;
    rotMatrix[1][0] = sinAngle;
    rotMatrix[1][1] = cosAngle;
}

__inline__ __device__ void Math::inverseRotationMatrix(float angle, Matrix& rotMatrix)
{
    float sinAngle = __sinf(angle*DEG_TO_RAD);
    float cosAngle = __cosf(angle*DEG_TO_RAD);
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
    if (length > FP_PRECISION) {
        vec = { vec.x / length, vec.y / length };
    }
    else {
        vec = { 1.0, 0.0 };
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

__host__ __device__ __inline__ float Math::length(float2 const & v)
{
    return sqrt(v.x * v.x + v.y * v.y);
}

__host__ __device__ __inline__ float Math::lengthSquared(float2 const & v)
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
        angleDiff -= 360.0;
    }
    if (angleDiff < 0.0) {
        angleDiff += 360.0;
    }
    return angleDiff;
}

__inline__ __device__ float
Math::calcDistanceToLineSegment(float2 const& startSegment, float2 const& endSegment, float2 const& pos, int const& boundary)
{
    auto const relPos = pos - startSegment;
    auto segmentDirection = endSegment - startSegment;
    if (length(segmentDirection) < FP_PRECISION) {
        return boundary + 1;
    }

    auto const segmentLength = length(segmentDirection);
    segmentDirection = segmentDirection / segmentLength;
    auto normal = segmentDirection;
    rotateQuarterCounterClockwise(normal);
    auto const signedDistanceFromLine = dot(relPos, normal);
    if (abs(signedDistanceFromLine) > boundary) {
        return boundary + 1;
    }

    auto const signedDistanceFromStart = dot(relPos, segmentDirection);
    if (signedDistanceFromStart < 0 || signedDistanceFromStart > segmentLength) {
        return boundary + 1;
    }

    return abs(signedDistanceFromLine);
}
