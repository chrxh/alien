#pragma once

#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineTestData/TestDataFactory.h"

namespace std
{
    template <>
    struct hash<std::pair<uint64_t, uint64_t>>
    {
        size_t operator()(const std::pair<uint64_t, uint64_t>& p) const
        {
            auto hash1 = std::hash<uint64_t>{}(p.first);
            auto hash2 = std::hash<uint64_t>{}(p.second);
            return hash1 ^ (hash2 << 1);
        }
    };
}

class IntegrationTestFramework : public ::testing::Test
{
public:
    IntegrationTestFramework(IntVector2D const& universeSize = IntVector2D{1000, 1000});
    virtual ~IntegrationTestFramework();

protected:
    double getEnergy(Description const& data) const;

    bool approxCompare(double expected, double actual, float precision = 0.001f) const
    {
        return TestDataFactory::approxCompare(expected, actual, precision);
    }

    bool approxCompare(float expected, float actual, float precision = 0.001f) const
    {
        return TestDataFactory::approxCompare(expected, actual, precision);
    }

    bool approxCompare(RealVector2D const& expected, RealVector2D const& actual) const
    {
        return TestDataFactory::approxCompare(expected, actual);
    }

    bool approxCompare(std::vector<float> const& expected, std::vector<float> const& actual) const
    {
        return TestDataFactory::approxCompare(expected, actual);
    }

    bool approxCompareAngles(float expected, float actual, float precision = 0.001f) const
    {
        return TestDataFactory::approxCompareAngles(expected, actual, precision);
    }

    bool compare(Description left, Description right) const;
    bool compare(CellDescription left, CellDescription right) const;
    bool compare(ParticleDescription left, ParticleDescription right) const;

    SimulationFacade _simulationFacade;
    SimulationParameters _parameters;
};