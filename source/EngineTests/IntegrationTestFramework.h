#pragma once

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <memory>

#include <Base/Definitions.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Description.h>
#include <EngineInterface/SimulationParameters.h>

#include <EngineTestData/TestHelper.h>

class IntegrationTestFramework : public ::testing::Test
{
public:
    IntegrationTestFramework(IntVector2D const& worldSize = IntVector2D{1000, 1000});
    virtual ~IntegrationTestFramework();

    static void cleanupGlobalContext();

protected:
    double getEnergy(Description const& data) const;

    static std::vector<CellDescription> getCellsForCreature(Description const& data, uint64_t creatureId)
    {
        std::vector<CellDescription> result;
        for (auto const& cell : data._cells) {
            if (cell._creatureId.has_value() && cell._creatureId.value() == creatureId) {
                result.push_back(cell);
            }
        }
        return result;
    }

    static std::vector<CellDescription>& getCellsForCreatureMut(Description& data, uint64_t creatureId)
    {
        static std::vector<CellDescription> empty;
        return data._cells;
    }

    bool approxCompare(double expected, double actual, float precision = 0.001f) const { return TestHelper::approxCompare(expected, actual, precision); }

    bool approxCompare(float expected, float actual, float precision = 0.001f) const { return TestHelper::approxCompare(expected, actual, precision); }

    bool approxCompare(RealVector2D const& expected, RealVector2D const& actual) const { return TestHelper::approxCompare(expected, actual); }

    bool approxCompare(std::vector<float> const& expected, std::vector<float> const& actual) const { return TestHelper::approxCompare(expected, actual); }

    bool approxCompareAngles(float expected, float actual, float precision = 0.001f) const
    {
        return TestHelper::approxCompareAngles(expected, actual, precision);
    }

    bool compare(Description left, Description right) const;
    bool compare(CellDescription left, CellDescription right) const;
    bool compare(ParticleDescription left, ParticleDescription right) const;

    SimulationFacade _simulationFacade;
    SimulationParameters _parameters;

private:
    struct TestSuiteContext
    {
        SimulationFacade simulationFacade;
        
        void cleanup();
    };
    static TestSuiteContext _globalContext;
    IntVector2D _worldSize;
};