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
    __inline__ __device__ static void angleCorrection(float &angle);
    __inline__ __device__ static void rotateQuarterCounterClockwise(float2 &v);
    __inline__ __device__ static float angleOfVector(float2 const& v);   //0 DEG corresponds to (0,-1)
    __inline__ __device__ static float2 unitVectorOfAngle(float angle);
    __inline__ __device__ static void normalize(float2 &vec);
    __inline__ __device__ static float2 normalized(float2 vec);
    __inline__ __device__ static float dot(float2 const &p, float2 const &q);
    __inline__ __device__ static float2 minus(float2 const &p);
    __inline__ __device__ static float length(float2 const & v);
    __inline__ __device__ static float lengthSquared(float2 const & v);

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

__inline__ __device__ float2 operator*(float2 const& p, float m)
{
    return{ p.x * m, p.y * m };
}

__inline__ __device__ float2 operator/(float2 const& p, float m)
{
    return{ p.x / m, p.y / m };
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

__device__ __inline__ float2 Math::minus(float2 const &p)
{
    return{ -p.x, -p.y };
}

__device__ __inline__ float Math::length(float2 const & v)
{
    return sqrt(v.x * v.x + v.y * v.y);
}

__device__ __inline__ float Math::lengthSquared(float2 const & v)
{
    return v.x * v.x + v.y * v.y;
}
