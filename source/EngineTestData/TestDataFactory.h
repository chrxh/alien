#pragma once

#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "Base/Math.h"

class TestDataFactory
{
public:
    static bool approxCompare(double expected, double actual, float precision = 0.001f)
    {
        return approxCompare(toFloat(expected), toFloat(actual), precision);
    }

    static bool approxCompare(float expected, float actual, float precision = 0.001f)
    {
        auto absNorm = std::abs(expected) + std::abs(actual);
        if (absNorm < precision) {
            return true;
        }
        return std::abs(expected - actual) / absNorm < precision;
    }

    static bool approxCompare(RealVector2D const& expected, RealVector2D const& actual, float precision = 0.001f)
    {
        return approxCompare(expected.x, actual.x, precision) && approxCompare(expected.y, actual.y, precision);
    }

    static bool approxCompare(std::vector<float> const& expected, std::vector<float> const& actual, float precision = 0.001f)
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

    static bool approxCompareAngles(float expected, float actual, float precision = 0.001f)
    {
        return approxCompare(Math::getNormalizedAngle(expected - actual, -180.0f), 0.0f, precision);
    }
};
