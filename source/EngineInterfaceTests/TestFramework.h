#pragma once

#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "Base/Math.h"

class TestFramework : public ::testing::Test
{
public:
    bool approxCompare(double expected, double actual, float precision = 0.001f) const
    {
        return approxCompare(toFloat(expected), toFloat(actual), precision);
    }

    bool approxCompare(float expected, float actual, float precision = 0.001f) const
    {
        auto absNorm = std::abs(expected) + std::abs(actual);
        if (absNorm < precision) {
            return true;
        }
        return std::abs(expected - actual) / absNorm < precision;
    }

    bool approxCompare(RealVector2D const& expected, RealVector2D const& actual, float precision = 0.001f) const
    {
        return approxCompare(expected.x, actual.x, precision) && approxCompare(expected.y, actual.y, precision);
    }

    bool approxCompare(std::vector<float> const& expected, std::vector<float> const& actual, float precision = 0.001f) const
    {
        if (expected.size() != actual.size()) {
            return false;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!approxCompare(expected[i], actual[i], precision)) {
                return false;
            }
        }
        return true;
    }

    bool approxCompareAngles(float expected, float actual, float precision = 0.001f) const
    {
        return approxCompare(Math::getNormalizedAngle(expected - actual, -180.0f), 0.0f, precision);
    }
};
