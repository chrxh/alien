#pragma once

#include <algorithm>

#include <gtest/gtest.h>

#include <Base/Definitions.h>
#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/GenomeDesc.h>

class TestHelper
{
public:
    static bool approxCompare(double expected, double actual, float precision = 0.001f);
    static bool approxCompare(float expected, float actual, float precision = 0.001f);
    static bool approxCompare(RealVector2D const& expected, RealVector2D const& actual, float precision = 0.001f);
    static bool approxCompare(std::vector<float> const& expected, std::vector<float> const& actual, float precision = 0.001f);
    static bool approxCompareAngles(float expected, float actual, float precision = 0.001f);

    static bool compare(Desc left, Desc right);
    static bool compare(ObjectDesc left, ObjectDesc right);
    static bool compare(EnergyDesc left, EnergyDesc right);
    static bool compare(GenomeDesc left, GenomeDesc right);
};
