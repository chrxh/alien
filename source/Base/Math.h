#pragma once

#include "Definitions.h"

namespace Const
{
    constexpr float Pi = 3.14159265358979f;
    constexpr float DegToRad = Pi / 180.0f;
    constexpr float RadToDeg = 180.0f / Pi;
}

class Math
{
public:
    static float length(RealVector2D const& v);
    static float angleOfVector(RealVector2D const& v);
    static RealVector2D rotateQuarterCounterClockwise(RealVector2D v);
    static RealVector2D unitVectorOfAngle(double angleInDeg);
    static RealMatrix2D calcRotationMatrix(float angleInDeg);  //rotation is clockwise
};

RealVector2D operator*(RealMatrix2D const& m, RealVector2D const& v);
