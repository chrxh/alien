#pragma once

#include "Definitions.h"
#include "DllExport.h"

constexpr float pi = 3.14159265358979f;
constexpr float degToRad = pi / 180.0f;
constexpr float radToDeg = 180.0f / pi;

class Math
{
public:
    BASE_EXPORT static float length(RealVector2D const& v);  //0 DEG corresponds to (0,-1)
    BASE_EXPORT static float angleOfVector(RealVector2D const& v);  //0 DEG corresponds to (0,-1)
    BASE_EXPORT static RealVector2D unitVectorOfAngle(double angle);
};
