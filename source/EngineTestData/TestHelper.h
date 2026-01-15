#pragma once

#include <algorithm>

#include <gtest/gtest.h>

#include <Base/Definitions.h>
#include <Base/Math.h>

#include <EngineInterface/Description.h>

class TestHelper
{
public:
    static bool approxCompare(double expected, double actual, float precision = 0.001f);
    static bool approxCompare(float expected, float actual, float precision = 0.001f);
    static bool approxCompare(RealVector2D const& expected, RealVector2D const& actual, float precision = 0.001f);
    static bool approxCompare(std::vector<float> const& expected, std::vector<float> const& actual, float precision = 0.001f);
    static bool approxCompareAngles(float expected, float actual, float precision = 0.001f);

    static bool compare(Description left, Description right);
    static bool compare(ObjectDescription left, ObjectDescription right);
    static bool compare(EnergyDescription left, EnergyDescription right);
};
