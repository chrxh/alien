#pragma once

#include "Definitions.h"
#include "DllExport.h"

constexpr double pi = 3.14159265358979;
constexpr double degToRad = pi / 180.0;
constexpr double radToDeg = 180.0 / pi;

class Math
{
public:
    BASE_EXPORT static double length(RealVector2D const& v);  //0 DEG corresponds to (0,-1)
    BASE_EXPORT static double angleOfVector(RealVector2D const& v);  //0 DEG corresponds to (0,-1)
};
