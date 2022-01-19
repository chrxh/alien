#pragma once

#include "Definitions.h"
#include "DllExport.h"

namespace Const
{
    constexpr float Pi = 3.14159265358979f;
    constexpr float DegToRad = Pi / 180.0f;
    constexpr float RadToDeg = 180.0f / Pi;
}

class Math
{
public:
    BASE_EXPORT static float length(RealVector2D const& v);
    BASE_EXPORT static float angleOfVector(RealVector2D const& v);
    BASE_EXPORT static RealVector2D rotateQuarterCounterClockwise(RealVector2D v);
    BASE_EXPORT static RealVector2D unitVectorOfAngle(double angleInDeg);
    BASE_EXPORT static RealMatrix2D calcRotationMatrix(float angleInDeg);  //rotation is clockwise
};

BASE_EXPORT RealVector2D operator*(RealMatrix2D const& m, RealVector2D const& v);
